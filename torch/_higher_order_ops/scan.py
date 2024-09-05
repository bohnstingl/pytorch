# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _set_compilation_env,
    reenter_make_fx,
    unique_graph_id,
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

from .utils import _from_fun, create_fw_bw_graph


aten = torch._ops.ops.aten


def create_fw_bw_graph_combinefn(combine_fn, input, dim):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            fw_inputs = [
                # pytree.tree_map(_from_fun, x)
                aten.slice(
                    pytree.tree_map(
                        _from_fun,
                        # aten.slice(x, dim, 0, 1, 1),
                        x,
                    ),
                    dim,
                    0,
                    1,
                    1,
                )
                for x in itertools.chain(input, input)
            ]

            fw_outputs_true = pytree.tree_map(_from_fun, combine_fn(*fw_inputs))
            if any(not isinstance(out, torch.Tensor) for out in fw_outputs_true):
                raise RuntimeError(
                    "Expect outputs of combine_fn to only contains tensors. "
                    f"Got types {[type(out) for out in fw_outputs_true]}."
                )

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            fw_graph, joint_graph = create_fw_bw_graph(
                combine_fn, False, fw_inputs, fw_outputs_true
            )

        return fw_graph, joint_graph


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    assert len(args) == 2 * num_leaves
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    assert num_leaves == len(combined_leaves)
    return combined_leaves


def scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    init: pytree.PyTree = None,
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        init (torch.Tensor): The inital scan carry, a tensor, or nested pytree of tensors that
            represents the first output of the scan, default ``None``. The ``init`` is expected to have the
            same pytree structure and shape as the output tensors of ``combine_fn``.
            In case the ``init`` is ``None``, the first element of input is used as ``init``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        # Usage without the usage of ``init``
        cumsum = scan(add, x, dim)
        # This produces the output
        cumsum = [x0, add(x0, x1), add(x1, x2)]

        # Usage with the usage of ``init``
        cumsum = scan(add, x, dim, init=i0)
        # This produces the output
        cumsum = [i0, add(i0, x0), add(x0, x1)]


    """
    assert callable(combine_fn), "combine_fn must be a callable, but got {combine_fn}"
    assert isinstance(dim, int), "dim must be an int, but got {type(dim)}"

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Support Autograd

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _scan_op_wrapper(*args, **kwargs):
        return scan(*args, **kwargs)

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(_scan_op_wrapper, backend="eager", fullgraph=True)(
                combine_fn, input, dim=dim, reverse=reverse, init=init
            )

    init = [] if init is None else init

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_input, spec_input = pytree.tree_flatten(input)

    if reverse:
        leaves_input = [torch.flip(elem, [dim]) for elem in leaves_input]

    assert (
        len(leaves_init) > 0 or len(leaves_input) > 0
    ), "expected at least 1 init or input leaf"
    if len(leaves_init) > 0:
        shape = leaves_init[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)
        num_el = shape[dim]
        output_spec = spec_init

        assert all(
            isinstance(x, torch.Tensor) for x in leaves_init
        ), "If init leaves are provided, they must be a Tensor"
    else:
        shape = leaves_input[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)
        num_el = shape[dim]
        output_spec = spec_input

        # If no init is provided, take the first time step of input as the init
        # and crop it off the original input
        leaves_init = [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input]
        if num_el > 1:
            leaves_input = [aten.slice(elem, dim, 1, None, 1) for elem in leaves_input]
        else:
            leaves_input = []

    if len(leaves_input) > 0:
        assert all(
            isinstance(x, torch.Tensor) for x in leaves_input
        ), "If input leaves are provided, they must be a Tensor"

        assert all(
            x.shape[dim] > 0 for x in leaves_input
        ), "If input leaves are provided, the scan dimension must be > 0"

        out = combine_fn(
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
        )
        out_leaves, tree_out = pytree.tree_flatten(out)
        assert len(leaves_input) == len(
            out_leaves
        ), "The number of leaves of the pytree of the output of the operator needs to match the lenght of the pytree of the input"
        for in_l, out_l in zip(leaves_init, out_leaves):
            assert (
                in_l.shape == out_l.shape
            ), "The pytree of the output of the operator needs to match the pytree of the init"

    # Add the init back to the result_flat as the first element
    if len(leaves_input) > 0:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=output_spec,
            num_leaves=len(leaves_input),
        )

        result_flat = scan_op(combine_fn, leaves_input, leaves_init, dim)

        if reverse:
            result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

        return pytree.tree_unflatten(result_flat, output_spec)
    else:
        return pytree.tree_unflatten(leaves_init, output_spec)


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, input, init, dim):
        return super().__call__(combine_fn, input, init, dim)


scan_op = ScanOp()


def generic_scan(operator, input, init, dim=0):
    def _scan(input, init):
        """Perform scan on `elems` using `elems_init."""
        num_elems = input[0].shape[dim]
        ind = 0
        out = init

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        outs, idxs = zip(
            *[
                (
                    torch.zeros(
                        list(e.size())[:dim]
                        + [num_elems + 1]
                        + list(e.size())[dim + 1 :],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64),
                )
                for i, e in enumerate(init)
            ]
        )

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                o.scatter_(dim, idx * ind, x)

        # Store the inits in the outs matrix.
        # These are the first elements of the scan outputs
        store_out_in_outs(out, ind)

        while ind < num_elems:
            out = operator(
                *out,
                *[aten.slice(elem, dim, ind, ind + 1, 1) for elem in input],
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(out, ind + 1)

            ind += 1

        return outs

    if len(input) == 0:
        return []

    scans = _scan(input, init)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    input: List[torch.Tensor],
    init: List[torch.Tensor],
    dim: int,
):
    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.empty_like(
                x_init,
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x, x_init in itertools.chain(zip(input, init), zip(input, init))
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    assert len(outputs) == len(
        input
    ), f"expected combine_fn to return {len(input)} results but got {len(outputs)}"

    for i, si, o in zip(input, sample_inputs, outputs):
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )
        assert (
            si.shape == o_meta.shape
        ), "The pytree of the out of the operator needs to match the input pytree"

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, input, dim)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in input]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, input, init, dim):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, input, init, dim)


class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        dim,
        num_leaves,
        *ops,
    ):
        init = ops[:num_leaves]
        input = ops[num_leaves:]

        ctx._joint_graph = joint_graph
        ctx._dim = dim
        ctx._num_leaves = num_leaves
        ctx._num_elems = input[0].size()[dim] + 1

        with torch._C._AutoDispatchBelowAutograd():
            outs = scan_op(fw_graph, input, init, dim)
            # outs = generic_scan(fw_graph, input, init, dim)
            input_init = [
                torch.concatenate([ini, inp], dim=dim) for ini, inp in zip(init, input)
            ]
            ctx.save_for_backward(*(input_init + list(outs)))

            return tuple(outs)

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

        num_elems = ctx._num_elems
        num_leaves = ctx._num_leaves
        joint_graph = ctx._joint_graph
        dim = ctx._dim

        # Retrieve the forward inputs and the forward outputs
        operands_outs = ctx.saved_tensors
        input, outs = operands_outs[:num_leaves], operands_outs[num_leaves:]

        inp_flipped = [
            aten.slice(torch.flip(inp, [dim]), dim, 0, -1, 1) for inp in input
        ]
        ones_inp = [torch.ones_like(aten.slice(inp, dim, 0, 1, 1)) for inp in input]
        out_flipped = [
            aten.slice(torch.flip(out, [dim]), dim, 1, None, 1) for out in outs
        ]

        helpers = joint_graph(
            *[aten.slice(fl, dim, 0, -1, 1) for fl in flat_grads],
            *out_flipped,
            *inp_flipped,
        )

        # This is the vector of 'first outputs'
        helper1 = torch.stack(
            [torch.concat([h, o], dim) for h, o in zip(helpers[num_leaves:], ones_inp)],
            0,
        )

        # First sub-diagonal containing the 'second outputs'
        helper2 = [
            torch.concat([o, h], dim) for h, o in zip(helpers[0:num_leaves], ones_inp)
        ]

        # More efficient version to compute the gradient matrix
        helper_mats = [
            torch.unsqueeze(
                torch.stack(
                    [
                        torch.concat(
                            [z] * (num_elems - n)
                            + [o]
                            + [aten.slice(h, dim, 1, n, 1) for h in helper2],
                            dim,
                        )
                        for o, z in zip(ones_inp, ones_inp)
                    ],
                    0,
                ),
                0,
            )
            for n in range(num_elems, 0, -1)
        ]
        helper_mats = torch.concat(helper_mats, 0)
        helper_mats = torch.cumprod(helper_mats, dim + 2)

        tril = torch.tril(
            torch.ones((num_elems, num_elems), device=helper_mats.device), diagonal=-1
        )
        # helper_mats is of shape num_elems x shape_of_input
        # shape_of_input contains num_elems at dim
        helper_mats = helper_mats - torch.reshape(
            tril,
            [num_elems]
            + [1]
            + [1] * (dim)
            + [num_elems]
            + [1] * (len(helper_mats[0].shape) - dim - 2),
        )

        # # Slow computation of matrix
        # # This is the matrix of 'second outputs'
        # helper_mats = torch.unsqueeze(
        #     torch.stack(
        #         [
        #             torch.concat([z] * (num_elems - 1) + [o], dim)
        #             for o, z in zip(ones_inp, zeros_inp)
        #         ],
        #         0,
        #     ),
        #     0,
        # )
        # for n in range(num_elems - 1, 0, -1):
        #     row = torch.stack(
        #         [
        #             hm * h2
        #             for hm, h2 in zip(
        #                 helper_mats[0],
        #                 [aten.slice(h, dim, n, n+1, 1) for h in helper2],
        #             )
        #         ],
        #         0,
        #     )
        #     row += torch.stack(
        #         [
        #             torch.concat([z] * (n - 1) + [o] + [z] * (num_elems - n), dim)
        #             for o, z in zip(ones_inp, zeros_inp)
        #         ],
        #         0,
        #     )
        #     helper_mats = torch.concat((torch.unsqueeze(row, 0), helper_mats), 0)

        # Elementwise matrix-vector multiplication to retrieve the final gradients
        grads = torch.split(
            torch.flip(torch.sum(helper1 * helper_mats, 0), [dim + 1]), 1, 0
        )
        grads_init = [aten.slice(torch.squeeze(g, 0), dim, 0, 1, 1) for g in grads]
        grads = [aten.slice(torch.squeeze(g, 0), dim, 1, None, 1) for g in grads]

        return None, None, None, None, *tuple(grads_init + grads)


@scan_op.py_impl(DispatchKey.Autograd)
def scan_autograd(combine_fn, input, init, dim):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (input,),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, input, init, dim)

    (
        fw_graph,
        joint_graph,
    ) = create_fw_bw_graph_combinefn(combine_fn, input, dim)

    num_leaves = len(init)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        dim,
        num_leaves,
        *(init + input),
    )
    return flat_out


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, input, init, dim):
    return trace_scan(mode, scan_op, combine_fn, input, init, dim)


@scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, init, dim):
    with mode:
        return combine_fn(*input, *input)


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, input, init, dim):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_init = ctx.unwrap_tensors(init)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        ret = scan_op(functional_combine_fn, unwrapped_input, unwrapped_init, dim)
    return ctx.wrap_tensors(ret)
