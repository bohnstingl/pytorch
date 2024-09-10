# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, List, Tuple

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode
from .utils import _from_fun, create_fw_bw_graph, _maybe_reenter_make_fx


aten = torch._ops.ops.aten


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
    
    # Create flat output and unflatten the output in wrapper
    return [*carry_flat, *combined_flat]


def _extract_carry_and_out(flat_out: List[Any], num_carry: int):
    return flat_out[:num_carry], flat_out[num_carry:]


def create_fw_bw_graph_combinefn(combine_fn, init, input, dim):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py
    
    def wrapper_combine_fn(*args):
        new_carry, y = _extract_carry_and_out(combine_fn(*args), len(init))
        return [*new_carry, *new_carry, *y]

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            num_init = len(init)
            
            fw_init = [pytree.tree_map(_from_fun, x) for x in init]
            fw_input = [pytree.tree_map(_from_fun, x.select(dim, 0)) for x in input]

            carry, outs = _extract_carry_and_out(wrapper_combine_fn(*fw_init, *fw_input), num_init)
            fw_carry, fw_outputs = [pytree.tree_map(_from_fun, c) for c in carry], [pytree.tree_map(_from_fun, o) for o in outs]
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

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            fw_graph = _maybe_reenter_make_fx(wrapper_combine_fn)(*fw_init, *fw_input)
            _, joint_graph = create_fw_bw_graph(
                combine_fn, False, (*fw_init, *fw_input), (*fw_carry, *fw_outputs[num_init:])
            )

        return fw_graph, joint_graph


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
            each tensor leaf is a stacked output along dim, where each slice is the output of a scan iteration.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

        i0 = torch.zeros(1)
        xs = torch.arange(1, 5)
        # returns torch.tensor([10]), torch.tensor([1., 3., 6., 10.])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if not isinstance(reverse, bool):
        raise RuntimeError("Reverse must be a bool, but got " + str(type(reverse)))

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(scan, backend="eager", fullgraph=True)(
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
    if any(x.shape[dim] == 0 for x in leaves_xs):
        raise RuntimeError("All xs leaves must have a scan dimension > 0")

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
            scan_op(combine_fn, leaves_init, leaves_xs, dim, reverse, None),
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

    def __call__(self, combine_fn, init, xs, dim, reverse, additional_inputs=None):
        additional_inputs = additional_inputs if additional_inputs is not None else []
        return super().__call__(combine_fn, init, xs, dim, reverse, additional_inputs)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, reverse=False, additional_inputs=None):
    additional_inputs = additional_inputs or []
    
    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        
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
        outs, outs_idxs = zip(
            *[
                [
                    torch.zeros(
                        list(e.size())[:dim] + [num_elems] + list(e.size())[dim:],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64).unsqueeze(dim),
                ]
                for i, e in enumerate(dummy_out)
            ]
        )
        
        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, outs_idxs):
                # o: (M, N, num_elems, ...)
                # x: (M, N, ...) -> (M, N, 1, ...)
                # ind * idx: (M, N, 1, ...) with values to be ind
                o.scatter_(dim, ind * idx, x.unsqueeze(dim))

        def cond(i, n, r):
            if (r and i < 0) or (not r and i > (n - 1)):
                return False
            else:
                return True

        def op(i):
            if reverse:
                return i - 1
            else:
                return i + 1

        ind = 0 if not reverse else num_elems - 1
        while cond(ind, num_elems, reverse):
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

            ind = op(ind)

        return [*carry, *list(outs)]

    scans = _scan(init, xs)
    return scans


def make_expanded_output_shape(dim, scan_length, shapes, use_sh=False):
    expanded_shapes = [
        tuple(
            (s if use_sh else -1) if i != dim else scan_length for i, s in enumerate(sh)
        )
        for sh in shapes
    ]
    return expanded_shapes


def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)


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
        fake_out_shapes = make_expanded_output_shape(
            dim, scan_length, [o.meta["val"].size() for o in output]
        )

        def expand_tensor(t, sh):
            if isinstance(t, torch.Tensor):
                return t.expand(*sh)
            return t

        expanded_outs = [
            pytree.tree_map(expand_tensor, t.meta["val"], sh)
            for t, sh in zip(output, fake_out_shapes)
        ]
        out = [*init, *expanded_outs]
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, dim, reverse, additional_inputs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, dim, reverse, additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        additional_inputs,
        *ops,
    ):
        init = ops[:num_leaves_init]
        xs = ops[num_leaves_init:]

        ctx._joint_graph = joint_graph
        ctx._dim = dim
        ctx._reverse = reverse
        ctx._num_leaves_init = num_leaves_init
        ctx._num_leaves_xs = len(xs)

        with torch._C._AutoDispatchBelowAutograd():
            carry, carries_outs = _extract_carry_and_out(scan_op(fw_graph, init, xs, dim, reverse, additional_inputs), num_leaves_init)
            
            # Collect the carries for each time step from the outs
            carries = carries_outs[:num_leaves_init]
            outs = carries_outs[num_leaves_init:]
            
            ctx.save_for_backward(*(init + xs + tuple(carries) + tuple(additional_inputs)))
            
            if reverse:
                carries_augmented = [torch.cat([torch.unsqueeze(i, dim), c[1:]], dim=dim) for i, c in zip(init, carries)]
                xs = [torch.flip(x, [dim]) for x in xs]
            else:
                carries_augmented = [torch.cat([torch.unsqueeze(i, dim), c[:-1]], dim=dim) for i, c in zip(init, carries)]
                
            init_bwd = [torch.ones_like(i) for i in init]
            xs_bwd = (*[torch.ones_like(y) for y in outs], 
                         *carries_augmented, 
                         *xs)
            g_init, g_outs = _extract_carry_and_out(scan_op(joint_graph, init_bwd, xs_bwd, dim, True, additional_inputs), num_leaves_init)
            
            if reverse:
                g_outs = [torch.flip(g, [dim]) for g in g_outs]
            
            # print([(e, e.shape) for e in (*g_init, *g_outs)])
            
            return (*carry, *outs)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by factorizing the components of the chainrule into
        a elementwise multiplcation of a matrix and a vector.
        The rows of the matrix can be efficiently computed using ``cumprod``.

        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or anested pytree of tensors.

        Example::

            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                return x + y

            The ``joint_graph`` g(.,.), used in the backward function, is the gradient of the function f(.,.).
            It computes the gradients for x and y of f. For example for the function f above
            def g(x: torch.Tensor, y: torch.Tensor):
                return 1., 1.
            In other words, the first output of g represents df(x,y)/dx, while the second one represents df(x,y)/dy.
            This will be exploited in the algorithm below.

            The inputs to ``scan`` in the forward path are x_1, x_2, ..., x_T
            The outputs of ``scan`` in the forward path are y_1, y_2, ..., y_T, where
            y_1 = x_1
            y_2 = f(y_1, x_2)
            ...
            y_T = f(y_{T-1}, x_T)

            The gradients of y_T with respect to the vector x are computed as:
            dy_T / dx = dy_T/dx_1 + dy_T/dx_2 + ... + dy_T/dx_T

            A few examples:
            dy_T/dx_T = df(y_{T-1}, x_T)/dx_T -> second output of g(y_{T-1}, x_T)

            dy_T/dx_{T-1} = df(y_{T-1}, x_T)/dy_{T-1} . df(y_{T-2}, x_{T-1})/dx_{T-1}
                          -> first output of g(y_{T-1}, x_T) . second output of g(y_{T-2}, x_{T-1})

            dy_T/dx_{T-2} = df(y_{T-1}, x_T)/dy_{T-1}
                            . df(y_{T-2}, x_{T-1})/dy_{T-2}
                            . df(y_{T-3}, x_{T-2})/dx_{T-2}
                          ->  first output of g(y_{T-1}, x_T)
                            . first output of g(y_{T-2}, x_{T-1})
                            . second output of g(y_{T-3}, x_{T-2})

            A conceptually similar pattern can be observerd for dy_{T-1} / dx
            dy_{T-1}/dx_T = 0

            dy_{T-1}/dx_{T-1} = df(y_{T-2}, x_{T-1})/dx_{T-1} -> second output of g(y_{T-2}, x_{T-1})

            dy_{T-1}/dx_{T-2} = df(y_{T-2}, x_{T-1})/dy_{T-2} . df(y_{T-3}, x_{T-2})/dx_{T-2}
                              -> first output of g(y_{T-2}, x_{T-1})
                              . second output of g(y_{T-3}, x_{T-2})

            If one inspects the pattern carefully, it becomes aparant that there is a product of
            'first outputs', followed by the last term which is a 'second output'.
            This can be represented with a matrix-vector multiplication, where the rows of the matrix contain
            the products of the 'first ouputs' and the vector contains the 'second outputs'.
            Furthermore, the product of 'first outputs' is continuously expanded leftwards with
            additional time steps. Therefore, the products can also be computed utilizing cumprod.
            The final gradients can be computed using an elementwise matrix-vector multiplication.
        """

        joint_graph = ctx._joint_graph
        dim = ctx._dim
        num_leaves_init = ctx._num_leaves_init
        num_leaves_xs = ctx._num_leaves_xs
        reverse = ctx._reverse
        
        # Retrieve the forward inputs and the forward outputs
        operands_outs = ctx.saved_tensors
        init = operands_outs[:num_leaves_init]
        xs = operands_outs[num_leaves_init:num_leaves_init+num_leaves_xs]
        carries = operands_outs[num_leaves_init+num_leaves_xs:2*num_leaves_init+num_leaves_xs]
        additional_inputs = operands_outs[2*num_leaves_init+num_leaves_xs:]

        with torch._C._AutoDispatchBelowAutograd():
            g_carry = flat_grads[:num_leaves_init]
            g_ys = flat_grads[num_leaves_init:]
            
            # g_ys_rearranged = [torch.cat([torch.unsqueeze(g, 0) for g in torch.tensor_split(g_y, num_elems, dim=dim)], dim=0) for g_y in g_ys]
            # carries_rearranged = [torch.cat([torch.unsqueeze(i, 0), c[:-1]], dim=0) for i, c in zip(init, carries)]
            
            # if reverse:
            #     input_rearranged = [torch.flip(torch.cat([torch.unsqueeze(i, 0) for i in torch.tensor_split(inp, num_elems, dim=dim)], dim=0), [0]) for inp in input]
            # else:
            #     input_rearranged = [torch.cat([torch.unsqueeze(i, 0) for i in torch.tensor_split(inp, num_elems, dim=dim)], dim=0) for inp in input]
            
            # input_bwd = (*g_ys_rearranged, 
            #              *carries_rearranged, 
            #              *input_rearranged)
            # g_init, g_outs = scan_op(joint_graph, init=g_carry, input=input_bwd, dim=0, reverse=True)
            
            # if reverse:
            #     g_outs = [torch.flip(g, [0]) for g in g_outs]
            
            # g_init = [g[-1, :] for g in g_init]
            # g_outs = [torch.cat([torch.squeeze(go, (0, 1)) for go in torch.tensor_split(go, ctx._num_elems, dim=0)], dim=dim) for go in g_outs]
            
            # # print([(e, e.shape) for e in (*g_init, *g_outs)])
            if reverse:
                carries_augmented = [torch.cat([torch.unsqueeze(i, dim), c[1:]], dim=dim) for i, c in zip(init, carries)]
                xs = [torch.flip(x, [dim]) for x in xs]
            else:
                carries_augmented = [torch.cat([torch.unsqueeze(i, dim), c[:-1]], dim=dim) for i, c in zip(init, carries)]
            
            xs_bwd = (*g_ys, 
                      *carries_augmented, 
                      *xs)
            g_init, g_outs = _extract_carry_and_out(scan_op(joint_graph, g_carry, xs_bwd, dim, True, additional_inputs), num_leaves_init)
            
            if reverse:
                g_outs = [torch.flip(g, [dim]) for g in g_outs]
            
            # print([(e, e.shape) for e in (*g_init, *g_outs)])

        return None, None, None, None, None, None, *g_init, *g_outs


@scan_op.py_impl(DispatchKey.Autograd)
def scan_autograd(combine_fn, init, input, dim, reverse, additional_inputs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (init, input),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, input, dim, reverse, additional_inputs)

    num_leaves_init = len(init)
    
    (
        fw_graph,
        joint_graph,
    # ) = create_fw_bw_graph_combinefn(combine_fn.keywords['combine_fn'], init, input, dim)
    ) = create_fw_bw_graph_combinefn(combine_fn, init, input, dim)
    # ) = create_fw_bw_graph_combinefn(wrapper_combine_fn, init, input, dim)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        additional_inputs,
        *(init + input),
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
                *[torch.select_copy(inp, dim, 0) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        out = (
            *carry,
            *tuple(
                t.unsqueeze(dim)
                .repeat(*([1] * dim + [scan_length] + [1] * (t.ndim - dim)))
                .clone()
                for t in outputs
            ),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, dim, reverse, additional_inputs):
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            torch.select_copy(inp, dim, 0) for inp in unwrapped_xs
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
                "Combine_fn might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input!"
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
        torch.stack([e[leave_ind] for e in op(result_flat)], dim)
        for leave_ind in range(num_leaves)
    ]
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
