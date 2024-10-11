# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, List, Tuple

import torch
import torch._dynamo.variables
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _has_potential_branch_output_alias,
    _set_compilation_env,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode

from .utils import _from_fun, _maybe_reenter_make_fx, create_fw_bw_graph


aten = torch._ops.ops.aten


# Helper functions that are also used from other places
def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)


def _extract_carry_and_out(flat_out: List[Any], num_carry: int):
    return list(flat_out[:num_carry]), list(flat_out[num_carry:])


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


# say we have a tensor of shape [3, 4, 5, 6]
# shift_source_dim_to_target_dim(t, 0, 3) -> [4, 5, 6, 3]
def shift_source_dim_to_target_dim(t, from_dim: int, to_dim: int):
    assert to_dim >= 0 and to_dim < t.ndim
    assert from_dim >= 0 and from_dim < t.ndim
    dims = list(range(0, t.ndim))
    dims.pop(from_dim)
    dims.insert(to_dim, from_dim)
    return t.permute(*dims)


def get_gradient_mask(tensor_list):
    return [True if v is not None and v.requires_grad else False for v in tensor_list]


def mask_gradient(grads, mask):
    return [g for g, m in zip(grads, mask) if m]


# Internal functions for scan.py
def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves)
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    carry, combined = combine_fn(carry, xs)
    carry_flat = pytree.tree_leaves(carry)
    combined_flat = pytree.tree_leaves(combined)
    assert num_init_leaves == len(carry_flat)
    return [*carry_flat, *combined_flat]


def create_fw_bw_graph_combinefn(combine_fn, init, xs, dim, additional_inputs):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are
    # detached in order not to raise problems with the function aliasing outputs

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            num_init = len(init)
            num_additional_inputs = len(additional_inputs)

            fw_init = [pytree.tree_map(_from_fun, x) for x in init]
            fw_xs = [pytree.tree_map(_from_fun, x).select(dim, 0) for x in xs]
            fw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]
            bw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]

            # TODO: do less re-computation with min-cut partitioner.
            def wrapper_fwd_combine_fn(*args):
                new_carry, y = _extract_carry_and_out(combine_fn(*args), num_init)
                return [*new_carry, *[n_c.clone().detach() for n_c in new_carry], *y]

            # The forward graph needs to be constructed from a different combine_fn than the joint_graph
            fw_graph = _maybe_reenter_make_fx(wrapper_fwd_combine_fn)(
                *fw_init, *fw_xs, *fw_additional_inputs
            )

            # Get gradient masks of inits, xs and additional_inputs.
            # These masks are used during the backward path to
            # return gradients for inits with requires_grad=True and None for the others
            init_mask = get_gradient_mask(init)
            xs_mask = get_gradient_mask(xs)
            additional_inputs_mask = get_gradient_mask(additional_inputs)

            carry, outs = _extract_carry_and_out(
                wrapper_fwd_combine_fn(
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                num_init,
            )

            fw_carry, fw_outputs = [pytree.tree_map(_from_fun, c) for c in carry], [
                pytree.tree_map(_from_fun, o) for o in outs
            ]
            if any(carry.shape != ini.shape for carry, ini in zip(fw_carry, init)):
                raise RuntimeError(
                    "Expect carry produced by combine_fn to only contains tensors. "
                    f"Got types {[type(carry) for carry in fw_carry]}."
                )
            if any(not isinstance(carry, torch.Tensor) for carry in fw_carry):
                raise RuntimeError(
                    "Expect carry produced by combine_fn to only contains tensors. "
                    f"Got types {[type(carry) for carry in fw_carry]}."
                )
            if any(not isinstance(out, torch.Tensor) for out in fw_outputs):
                raise RuntimeError(
                    "Expect outputs produced by combine_fn to only contains tensors. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            # The joint graph is constructed with the requires_grad forced to True for the init, xs and additional_inputs
            # This is necessary because during the backward scan, we need the g_init for the other gradients, even if
            # we don't directly need g_init
            _, joint_graph = create_fw_bw_graph(
                combine_fn,
                False,
                (
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                (*fw_carry, *fw_outputs[num_init:]),
            )

            g_c, g_xs = _extract_carry_and_out(
                joint_graph(
                    *fw_carry,
                    *fw_outputs[num_init:],
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                num_init,
            )

            # Check whether the init and the carries have the same requires_grad flags
            # Scan enforces that the levaes of inits and of the carries require the same gradients
            if any(
                cm is not im or g_cm is not im
                for g_cm, cm, im in zip(
                    get_gradient_mask(g_c), get_gradient_mask(carry), init_mask
                )
            ):
                raise RuntimeError(
                    "The init and carries need to have the same require_grad structure! \
                    E.g., check for `with torch.no_gard` statements in the combine_fn."
                )

            # The gradient masks are combined with the initial ones.
            # The reason is that the combine_fn might contain ``with torch.no_grad()`` statements
            # Thus, even if the gradients of xs or additional_inputs should be tracked,
            # The ``torch.no_grad()`` statements may break the gradient tracking
            xs_mask = xs_mask and get_gradient_mask(
                g_xs[: len(g_xs) - num_additional_inputs]
            )
            additional_inputs_mask = additional_inputs_mask and get_gradient_mask(
                g_xs[len(g_xs) - num_additional_inputs :]
            )

            def wrapper_bwd_combine_fn(*args):
                carried_g_additional_input = args[:num_additional_inputs]

                g_c, g_xs = _extract_carry_and_out(
                    joint_graph(*args[num_additional_inputs:]), num_init
                )
                current_g_additional_inputs = g_xs[len(g_xs) - num_additional_inputs :]

                new_g_additional_inputs = [
                    # The clone().detach() is required to avoid aliasing inputs
                    carr_g + curr_g if add_inp_m else carr_g.clone().detach()
                    for add_inp_m, carr_g, curr_g in zip(
                        additional_inputs_mask,
                        carried_g_additional_input,
                        current_g_additional_inputs,
                    )
                ]
                g_xs = g_xs[: len(g_xs) - num_additional_inputs]

                # We need to mask the g_xs so that no None values are returned
                # The reason being that in the backward implementation, we store
                # The gradients in a matrix and thus, None values are problematic
                g_xs = mask_gradient(g_xs, xs_mask)
                g_c = [
                    g if g_m else torch.zeros_like(gi)
                    for g, g_m, gi in zip(
                        g_c,
                        init_mask,
                        args[num_additional_inputs : num_additional_inputs + num_init],
                    )
                ]

                return [*new_g_additional_inputs, *g_c, *g_xs]

        new_joint_graph = _maybe_reenter_make_fx(wrapper_bwd_combine_fn)(
            *bw_additional_inputs,
            *fw_carry,
            *fw_outputs[num_init:],
            *fw_init,
            *fw_xs,
            *fw_additional_inputs,
        )
        return fw_graph, new_joint_graph, init_mask, xs_mask, additional_inputs_mask


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], Tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> Tuple[pytree.PyTree, pytree.PyTree]:
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The inital scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves): The input tensor, or nested pytree of tensors.

    Kwargs:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if not isinstance(reverse, bool):
        raise RuntimeError("Reverse must be a bool, but got " + str(type(reverse)))

    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.
    # TODO: The dim argument can alternatively be handled by always moving this dim
    # to zero, scan over the 0-th dim and then move the dim back for the results

    if not torch._dynamo.is_compiling():
        from torch._dynamo.backends.debugging import (
            make_eager_backend_with_torch_function_mode,
        )

        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"
                return torch.compile(scan, backend=backend, fullgraph=True)(
                    combine_fn, init, xs, dim=dim, reverse=reverse
                )

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs, spec_xs = pytree.tree_flatten(xs)

    if len(leaves_init) == 0:
        raise RuntimeError("Init tensors must be provided")
    if any(not isinstance(x, torch.Tensor) for x in leaves_init):
        raise RuntimeError("All init leaves must be a Tensor")
    if any(not isinstance(x, torch.Tensor) for x in leaves_xs):
        raise RuntimeError("All xs leaves must be a Tensor")
    if any(x.ndim < dim for x in leaves_xs):
        raise RuntimeError(
            "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
        )
    if any(x.shape[dim] == 0 for x in leaves_xs):
        raise RuntimeError(
            "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
        )

    if len(leaves_xs) > 0:
        shape = leaves_xs[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)

        out = combine_fn(
            pytree.tree_unflatten(leaves_init, spec_init),
            pytree.tree_unflatten([elem.select(dim, 0) for elem in leaves_xs], spec_xs),
        )

        # The first output needs to have the same pytree as init
        carry_leaves = pytree.tree_leaves(out[0])
        if len(carry_leaves) != len(leaves_init):
            raise RuntimeError(
                "The number of leaves of the pytree of the new carry produced by the operator\
 needs to match the length of the pytree of the init"
            )
        if any(
            in_l.shape != out_l.shape for in_l, out_l in zip(leaves_init, carry_leaves)
        ):
            raise RuntimeError(
                "The pytree of the new carry produced by the operator needs to match the pytree of the init"
            )

        # There are no pytree restrictions on the second output of the operator
        _, tree_out = pytree.tree_flatten(out[1])

        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec_init=spec_init,
            spec_xs=spec_xs,
            num_init_leaves=len(leaves_init),
            num_inp_leaves=len(leaves_xs),
        )

        result_carry, result_flat = _extract_carry_and_out(
            scan_op(
                combine_fn, leaves_init, leaves_xs, dim, reverse, additional_inputs=[]
            ),
            len(leaves_init),
        )

        return pytree.tree_unflatten(result_carry, spec_init), pytree.tree_unflatten(
            result_flat, tree_out
        )

    else:
        return pytree.tree_unflatten(leaves_init, spec_init), xs


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, xs, dim, reverse, additional_inputs):
        return super().__call__(combine_fn, init, xs, dim, reverse, additional_inputs)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, reverse=False, additional_inputs=None):
    additional_inputs = additional_inputs if additional_inputs is not None else []

    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        if reverse:
            ind = num_elems - 1
        else:
            ind = 0

        # Compute dummy shapes for the pre-allocation
        num_init_leaves = len(init)
        dummy_carry, dummy_out = _extract_carry_and_out(
            operator(
                *carry,
                *[first_slice_copy(elem, dim) for elem in xs],
                *additional_inputs,
            ),
            num_init_leaves,
        )

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (num_elems, M, N, ...)
        # idx: (1, M, N)
        outs = [
            torch.zeros(
                [num_elems] + list(e.size()),
                dtype=e.dtype,
                device=e.device,
            )
            for i, e in enumerate(dummy_out)
        ]
        idxs = [
            torch.ones_like(e, dtype=torch.int64).unsqueeze(0)
            for i, e in enumerate(dummy_out)
        ]

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (num_elems, M, N ...)
                # x: (M, N, ...) -> (1, M, N)
                # ind * idx: (1, M, N,) with values to be ind
                # essentially: o[ind][n][k] = x[0][n][k]
                o.scatter_(0, ind * idx, x.unsqueeze(0))

        for i in range(num_elems):
            ind = i if not reverse else num_elems - i - 1
            carry, out = _extract_carry_and_out(
                operator(
                    *carry,
                    *[elem.select(dim, ind) for elem in xs],
                    *additional_inputs,
                ),
                num_init_leaves,
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(out, ind)

        return [*carry, *list(outs)]

    scans = _scan(init, xs)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: List[torch.Tensor],
    xs: List[torch.Tensor],
    dim: int,
    reverse: bool,
    additional_inputs: List[torch.Tensor],
):
    with disable_proxy_modes_tracing():
        sample_inits = [x_init.clone() for x_init in init]
        sample_inputs = [first_slice_copy(x, dim) for x in xs]
        sample_additional_inputs = [x.clone() for x in additional_inputs]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_inits, *sample_inputs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None

    carry, output = _extract_carry_and_out(outputs, len(init))

    for ini, ca in zip(init, carry):
        ini_meta = ini
        carry_meta = ca.meta["tensor_meta"]
        carry_val = ca.meta["val"]
        if (
            carry_val.device != ini_meta.device
            or carry_meta.dtype != ini_meta.dtype
            or carry_meta.shape != ini_meta.shape
        ):
            raise RuntimeError(
                f"Expected metadata of the combine_fn result {carry_meta} to be the same as "
                + f"the metadata of init with {ini_meta}"
            )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, dim, reverse, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[dim]
        fake_carry, fake_outputs = _extract_carry_and_out(
            [o.meta["val"] for o in outputs], len(init)
        )
        out = (
            *fake_carry,
            *(stack_y(t, scan_length) for t in fake_outputs),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, dim, reverse, additional_inputs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, dim, reverse, additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def extract_init_xs_additional_inputs(flat_args, num_leaves_init, num_leaves_xs):
        init = flat_args[:num_leaves_init]
        xs = flat_args[num_leaves_init : num_leaves_init + num_leaves_xs]
        additional_inputs = flat_args[num_leaves_init + num_leaves_xs :]
        return init, xs, additional_inputs

    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        num_leaves_xs,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
        *flat_args,
    ):
        ctx._joint_graph = joint_graph
        ctx._dim = dim
        ctx._reverse = reverse
        ctx._num_leaves_init = num_leaves_init
        ctx._num_leaves_xs = num_leaves_xs
        init, xs, additional_inputs = ScanAutogradOp.extract_init_xs_additional_inputs(
            list(flat_args), num_leaves_init, num_leaves_xs
        )
        ctx._num_additional_inputs = len(additional_inputs)

        ctx._carry_mask = carry_mask
        ctx._xs_mask = xs_mask
        ctx._additional_inputs_mask = additional_inputs_mask

        with torch._C._AutoDispatchBelowAutograd():
            carry, carries_outs = _extract_carry_and_out(
                scan_op(fw_graph, init, xs, dim, reverse, additional_inputs),
                num_leaves_init,
            )

            # Collect the carries for each time step from the outs
            # and save them for the backward path
            carries = carries_outs[:num_leaves_init]
            outs = carries_outs[num_leaves_init:]
            ctx.save_for_backward(*(init + xs + additional_inputs + carries))
            ctx._num_leaves_ys = len(outs)
            return (*carry, *outs)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by constructing using an additional scan operator with the gradients

        Args:
            flat_grads (torch.Tensor): The tensor of flattened upstream gradients.

        Example::

            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                next_carry = y = x * y
                return next_carry, y

            The ``joint_graph`` g(.,.), used in the backward function, is the joint function of the function f(.,.).
            It receives the upstream gradients and the inputs of f and computes the gradients
            for x and y of f. For example for the function f above
            def g(g_new_carry: torch.Tensor, g_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
                return g_y * y + g_new_carry * y, g_y * x + g_new_carry * x

            To use a scan operation for the backward path as well, the function f is modified such that it
            returns all carries and not only the last one. In particular:
            def f_autograd(x: torch.Tensor, y: torch.Tensor):
                next_carry, y = f(x, y)
                return next_carry, (next_carry, y)

            The inputs to ``scan`` in the forward path are init; xs_1, xs_2, ..., xs_T
            With the modified function f, the outputs of ``scan`` in the forward path are (c_1, y_1), (c_2, y_2), ..., (c_T, y_T).
            The backward function receives gradients for c_T -> g_c_T and for y_1, y_2, ... y_T -> g_y_1, g_y_2, ... g_y_T = g_ys

            The gradients of init and xs can then be computed as
            xs_bwd = (*g_ys, *carries, *xs)
            g_init, g_xs = scan(joint_graph, g_c_T, xs_bwd, dim, True)

        """

        def prepare_xs_carries_for_bwd(xs, init, carries, dim, reverse):
            if reverse:
                return [torch.flip(x, [dim]) for x in xs], [
                    torch.cat(
                        [torch.unsqueeze(i, dim), torch.flip(c[1:], [dim])], dim=dim
                    )
                    for i, c in zip(init, carries)
                ]
            else:
                return xs, [
                    torch.cat([torch.unsqueeze(i, dim), c[:-1]], dim=dim)
                    for i, c in zip(init, carries)
                ]

        def prepare_final_gradients_xs(g_xs, dim, reverse):
            # The g_xs coming from the backward scan has the outputs always stacked at dim 0
            # Thus, first we shift the 0-th dim to the dim of the forward scan
            g_xs = [shift_source_dim_to_target_dim(g, 0, dim) for g in g_xs]

            # Second, if needed, we flip the g_xs along dim
            if reverse:
                g_xs = [torch.flip(g, [dim]) for g in g_xs]

            return g_xs

        def prepare_initial_gradients(
            flat_grads, additional_inputs, num_leaves_init, num_leaves_ys, dim
        ):
            # The flat gradients are a list of g_c_T, g_ys
            g_c_T, g_ys, _ = ScanAutogradOp.extract_init_xs_additional_inputs(
                list(flat_grads), num_leaves_init, num_leaves_ys
            )

            # In case the reverse flag is used, the upstream g_ys need to be flipped along dim
            if reverse:
                g_ys = [torch.flip(g, [dim]) for g in g_ys]

            # The initial gradients for the additional_inputs are all zeros
            g_additional_inputs = [torch.zeros_like(ai) for ai in additional_inputs]
            return g_c_T, g_ys, g_additional_inputs

        def mask_grads_with_None(real_grads, mask):
            g_list = []
            for m, g in zip(mask, real_grads):
                if m:
                    g_list.append(g)
                else:
                    g_list.append(None)
            return g_list

        def expand_grads_with_None(real_grads, mask):
            g_list = []
            real_grads_cnt = 0
            for m in mask:
                if m:
                    g_list.append(real_grads[real_grads_cnt])
                    real_grads_cnt += 1
                else:
                    g_list.append(None)
            return g_list

        joint_graph = ctx._joint_graph
        dim = ctx._dim
        reverse = ctx._reverse
        num_leaves_init = ctx._num_leaves_init
        num_leaves_xs = ctx._num_leaves_xs
        num_leaves_ys = ctx._num_leaves_ys

        carry_mask = ctx._carry_mask
        xs_mask = ctx._xs_mask
        num_xs_mask = sum(xs_mask)
        additional_inputs_mask = ctx._additional_inputs_mask
        num_additional_inputs = ctx._num_additional_inputs

        # The results from the forward scan are always stacked on dim 0
        # The gradients though need to be provided with the correct scan dimension dim
        # Therefore, the inputs to the backward scan are all on dim 0, and the scan is performed on dim 0
        # The gradient outputs are finally shifted at the end to the correct dim
        bwd_scan_dim = 0

        # Retrieve the forward inputs and the forward outputs
        flat_args = ctx.saved_tensors
        carries = flat_args[-num_leaves_init:]
        init, xs, additional_inputs = ScanAutogradOp.extract_init_xs_additional_inputs(
            list(flat_args[:-num_leaves_init]), num_leaves_init, num_leaves_xs
        )

        # The backward scan operates on the 0-th dim and thus the original inputs need to be
        # permuted accordingly
        xs = [
            shift_source_dim_to_target_dim(o, dim, bwd_scan_dim)
            for o in flat_args[num_leaves_init : num_leaves_init + num_leaves_xs]
        ]

        with torch._C._AutoDispatchBelowAutograd():
            # Prepare the initial gradients for the backward scan
            g_c_T, g_ys, g_additional_inputs = prepare_initial_gradients(
                list(flat_grads),
                additional_inputs,
                num_leaves_init,
                num_leaves_ys,
                bwd_scan_dim,
            )
            xs, carries = prepare_xs_carries_for_bwd(
                xs, init, carries, bwd_scan_dim, reverse
            )
            xs_bwd = [*g_ys, *carries, *xs]

            g_outs = scan_op(
                joint_graph,
                [*g_additional_inputs, *g_c_T],
                xs_bwd,
                bwd_scan_dim,
                True,
                additional_inputs,
            )
            new_g_additional_inputs = g_outs[:num_additional_inputs]
            g_init = g_outs[
                num_additional_inputs : num_additional_inputs + num_leaves_init
            ]
            g_xs = g_outs[len(g_outs) - num_xs_mask :]
            g_xs = prepare_final_gradients_xs(g_xs, dim, reverse)

        new_g_additional_inputs = mask_grads_with_None(
            new_g_additional_inputs, additional_inputs_mask
        )
        g_xs = expand_grads_with_None(g_xs, xs_mask)
        g_init = mask_grads_with_None(g_init, carry_mask)
        return *[None] * 9, *g_init, *g_xs, *new_g_additional_inputs


@scan_op.py_impl(DispatchKey.Autograd)
def scan_autograd(combine_fn, init, xs, dim, reverse, additional_inputs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    # TODO: Figure out how to do this in dispatcher so that we don't have to do this check here
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (init, xs, additional_inputs),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, xs, dim, reverse, additional_inputs)

    # TODO: The create_fw_bw is always invoked twice:
    # Once in the forward path and
    # once in the backward path, where it should only be invoked for the grad grad case.
    # We don't support this currently
    if not torch.is_grad_enabled():
        # This clause is hit in the case of double backward.
        # Currently scan does not support this and thus we just dummy call another scan
        # The scan dim in the backward backward is always zero, because the
        # scan outputs during the forward are always collected at dim=0
        bwd_dim = 0
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, xs, bwd_dim, reverse, additional_inputs)

    num_leaves_init = len(init)
    num_leaves_xs = len(xs)

    (
        fw_graph,
        joint_graph,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
    ) = create_fw_bw_graph_combinefn(combine_fn, init, xs, dim, additional_inputs)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        num_leaves_xs,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
        *(init + xs + additional_inputs),
    )
    return *flat_out[:num_leaves_init], *flat_out[num_leaves_init:]


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    return trace_scan(
        mode, scan_op, combine_fn, init, xs, dim, reverse, additional_inputs
    )


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    with mode:
        scan_length = xs[0].shape[dim]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[first_slice_copy(inp, dim) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        out = [
            *carry,
            *[stack_y(t, scan_length) for t in outputs],
        ]
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, dim, reverse, additional_inputs):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            first_slice_copy(inp, dim) for inp in unwrapped_xs
        ]
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        if _has_potential_branch_input_mutation(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be modifying the input! Please also check the gradient of combine_fn for modifying the input"
            )
        if _has_potential_branch_input_alias(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input! Please also check the gradient of combine_fn for modifying the input"
            )
        if _has_potential_branch_output_alias(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing its outputs! Please also check the gradient of combine_fn for aliasing its outputs"
            )
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            dim,
            reverse,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


# dense implementation for scan. Used for testing only.
def _fake_scan(combine_fn, init, xs=None, dim=0, reverse=False):
    carry_leaves, carry_spec = pytree.tree_flatten(init)
    inp_leaves, inp_spec = pytree.tree_flatten(xs)
    if xs is None or len(inp_leaves) == 0:
        return init, []
    result_flat = []
    carry = carry_leaves
    op = reversed if reverse else lambda x: x

    dummy_carry, dummy_out = combine_fn(
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(
            [first_slice_copy(elem, dim) for elem in inp_leaves],
            inp_spec,
        ),
    )
    dummy_out_leaves, dummy_out_spec = pytree.tree_flatten(dummy_out)
    num_leaves = len(dummy_out_leaves)

    for ind in op(range(inp_leaves[0].size(dim))):
        xs = [elem.select(dim, ind) for elem in inp_leaves]

        carry, y = combine_fn(
            pytree.tree_unflatten(carry, carry_spec),
            pytree.tree_unflatten(xs, inp_spec),
        )
        carry, _ = pytree.tree_flatten(carry)
        y, _ = pytree.tree_flatten(y)
        result_flat.append(y)

    results = [
        torch.stack([e[leave_ind] for e in op(result_flat)])
        for leave_ind in range(num_leaves)
    ]
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
