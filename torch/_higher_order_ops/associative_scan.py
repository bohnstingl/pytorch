# mypy: allow-untyped-defs
import contextlib
import functools
import itertools
from typing import Callable, List, Tuple

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import _add_batch_dim, get_unwrapped, maybe_get_bdim
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._dispatch.python import suspend_functionalization
from torch._guards import detect_fake_mode
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

def create_fw_bw_graph_combinefn(combine_fn, dim, *operands):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # fw_inputs = pytree.tree_map(_from_fun, operands)
            
            fw_inputs = [
                # pytree.tree_map(_from_fun, x[slice_along_axis(0, 1, stride=None, dim=dim)])
                # pytree.tree_map(_from_fun, x)[slice_along_axis(0, 1, stride=None, dim=dim)]
                pytree.tree_map(_from_fun, x)
                for x in itertools.chain(operands, operands)
            ]

            fw_outputs_true = pytree.tree_map(_from_fun, combine_fn(*fw_inputs))
            if any(
                not isinstance(out, torch.Tensor)
                for out in fw_outputs_true
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of true_fn to only contains tensors or None. "
                    f"Got types {[type(out) for out in fw_outputs_true]}."
                )

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            fw_graph, joint_graph = create_fw_bw_graph(
                combine_fn, False, (*fw_inputs,), fw_outputs_true
            )

        return fw_graph, joint_graph
    
# def create_fw_bw_graph_wrapfn(fn, combine_fn, spec, num_leaves, *operands):
def create_fw_bw_graph_wrapfn(fn, combine_fn, *operands):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # fw_inputs = pytree.tree_map(_from_fun, operands)
            
            fw_inputs = [
                # pytree.tree_map(_from_fun, x[slice_along_axis(0, 1, stride=None, dim=dim)])
                # pytree.tree_map(_from_fun, x)[slice_along_axis(0, 1, stride=None, dim=dim)]
                pytree.tree_map(_from_fun, x)
                for x in itertools.chain(operands, operands)
                # for x in operands
            ]

            # fw_outputs_true = pytree.tree_map(_from_fun, fn(combine_fn, spec, num_leaves, *fw_inputs))
            fw_outputs_true = pytree.tree_map(_from_fun, fn(combine_fn, fw_inputs))
            if any(
                not isinstance(out, torch.Tensor)
                for out in fw_outputs_true
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of true_fn to only contains tensors or None. "
                    f"Got types {[type(out) for out in fw_outputs_true]}."
                )

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            # fw_graph = _maybe_reenter_make_fx(fn)(combine_fn, spec, num_leaves, *fw_inputs)
            fw_graph = _maybe_reenter_make_fx(fn)(combine_fn, fw_inputs)

        return fw_graph
    
def create_fw_bw_graph_partial(fn, *operands):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # fw_inputs = pytree.tree_map(_from_fun, operands)
            
            fw_inputs = [
                # pytree.tree_map(_from_fun, x[slice_along_axis(0, 1, stride=None, dim=dim)])
                # pytree.tree_map(_from_fun, x)[slice_along_axis(0, 1, stride=None, dim=dim)]
                pytree.tree_map(_from_fun, x)
                for x in itertools.chain(operands, operands)
            ]

            fw_outputs_true = pytree.tree_map(_from_fun, fn(*fw_inputs))
            if any(
                not isinstance(out, torch.Tensor)
                for out in fw_outputs_true
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of true_fn to only contains tensors or None. "
                    f"Got types {[type(out) for out in fw_outputs_true]}."
                )

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            fw_graph = _maybe_reenter_make_fx(fn)(*fw_inputs)

        return fw_graph

def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
# def wrap_combine_fn_flat(combine_fn, spec, num_leaves, *args):
    if len(args) != 2 * num_leaves:
        raise ValueError(
            "The number of leaves provided to the combine wrapper needs to be twice the number of arguments"
        )
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    # combined = combine_fn(lhs, rhs)
    combined_leaves = combine_fn(lhs, rhs)
    # combined_leaves2 = combined_leaves
    # combined_leaves2 = [combined_leaves]
    combined_leaves = pytree.tree_leaves(combined_leaves)
    if num_leaves != len(combined_leaves):
        raise ValueError(
            "The number of levaes of the inputs need to be identical to the number of leaves of the scan output"
        )
    # return combined_leaves2
    return combined_leaves

def wrap_combine_fn_flat2(*args, combine_fn, spec, num_leaves):
    if len(args) != 2 * num_leaves:
        raise ValueError(
            "The number of leaves provided to the combine wrapper needs to be twice the number of arguments"
        )
    # combined = combine_fn(args[:num_leaves][0], args[num_leaves:][0])
    combined = combine_fn(args[0], args[1])
    # lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    # rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    # combined = combine_fn(lhs, rhs)
    return combined


def check_args(combine_fn, leaves, tree, dim):
    if not callable(combine_fn):
        raise ValueError("combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError("dim must be an int, but got {type(dim)}")

    if len(leaves) < 1:
        raise ValueError("expected at least 1 input leaf")

    if not all(isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError("input leaves must be a Tensor")
    shape = leaves[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    for x in leaves[1:]:
        if x.shape != shape:
            raise ValueError("All input tensors must have the same shape")

    out = combine_fn(
        pytree.tree_unflatten(leaves, tree),
        pytree.tree_unflatten(leaves, tree),
    )

    out_leaves, tree_out = pytree.tree_flatten(out)
    if tree.num_nodes != tree_out.num_nodes or any(
        [
            o.shape != i.shape or o.dtype != i.dtype or o.device != i.device
            for o, i in zip(out_leaves, leaves)
        ]
    ):
        raise ValueError(
            "The pytree of the output of the operator needs to match the input pytree"
        )


def _interleave(a, b, dim):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        pad = [0, 0] * b.ndim
        pad[
            (b.ndim - dim - 1) * 2 + 1
        ] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = interleaved[
            slice_along_axis(0, b.shape[dim] + a.shape[dim] - 1, dim=dim)
        ]
    return interleaved


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


def slice_along_axis(start, end, stride=None, dim=0):
    return (slice(None),) * dim + (slice(start, end, stride),)


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    generic_scan: bool = False,
    lifted_args: Tuple = (),
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative pointwise combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must satisfy the associativity property.
        input (Tuple of possibly nested dict/list/tuple of tensors): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): The dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to the dimension.
        generic_scan (bool): A boolean stating whether a generic scan mode should be used.
            If ``generic_scan=False``, ``combine_op`` must be pure and may only contain pointwise operations.
            Moreover, ``generic_scan=False`` may just be used on CUDA tensors.
            On the other hand, ``generic_scan=False`` should be more efficient than ``generic_scan=True``,
            whenever it can be used.
            Note: This argument is automatically computed internally, but ``generic_scan=True`` can be enforced
            Note: In case the output of `torch.associative_scan` is part of backward(),
            i.e., gradients need to propagate through `torch.associative_scan`,
            then ``generic_scan=True`` is required
        lifted_args (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    leaves, spec = pytree.tree_flatten(input)

    check_args(combine_fn, leaves, spec, dim)

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    if generic_scan:
        result_flat = generic_associative_scan(combine_fn, leaves, dim, spec, lifted_args)
    else:
        result_flat = associative_scan_op(combine_fn, leaves, dim, spec, lifted_args)

    if reverse:
        result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, elems_flat, dim=0, spec='*', lifted_args=()):
    # TODO: The recursion involved here "unrolls" the scan
    # function for all inputs. Could there be a more efficient
    # way instead of running over the operation in sequence?
    def _scan(elems):
        # num_elems = elems[0].shape[dim]

        # if num_elems < 2:
        #     return elems
        
        # while nelems >= 2:
            
        #     reduced_elems = operator(
        #         *[elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems],
        #         *[elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems],
        #         #*lifted_args,
        #     )
        
        #     # # lhs = [elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems][0]
        #     # lhs = pytree.tree_unflatten([elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems], spec)
        #     # # rhs = [elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems][0]
        #     # rhs = pytree.tree_unflatten([elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems], spec)
            
        #     # reduced_elems = operator(lhs, rhs)
            
        #     # reduced_elems = pytree.tree_leaves(reduced_elems)
        #     reduced_elems2 = [reduced_elems]
            
        #     nelems = 1
        
        # # return _scan([reduced_elems], 1)
        # return reduced_elems
        
        # # return operator(*elems, *elems)#, *())
    
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = operator(
            *[elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems],
            *[elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems],
            #*lifted_args,
        )
        
        # lhs = [elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems][0]
        # # lhs = pytree.tree_unflatten([elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems], spec)
        # rhs = [elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems][0]
        # # rhs = pytree.tree_unflatten([elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems], spec)
        
        # reduced_elems = operator(lhs, rhs)
        
        # # reduced_elems = pytree.tree_leaves(reduced_elems)
        # reduced_elems = [reduced_elems]
        
        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            # # lhs = pytree.tree_unflatten([e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems], spec)
            # lhs = [e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems][0]
            # # rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems], spec)
            # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
            
            even_elems = operator(
                *[e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems],
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                # *lifted_args,
            )
            # even_elems = operator(
            #     lhs, rhs
            # )
            
            # # even_elems = pytree.tree_leaves(even_elems)
            # even_elems = [even_elems]
            
        else:
            # # lhs = pytree.tree_unflatten(odd_elems, spec)
            # lhs = odd_elems[0]
            # # rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],  spec)
            # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
            
            even_elems = operator(
                *odd_elems,
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                # *lifted_args,
            )
            
            # even_elems = operator(
            #     *odd_elems,
            #     *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
            #     # *lifted_args,
            # )
            
            # even_elems = [even_elems]
            # # even_elems = pytree.tree_leaves(even_elems)

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([elem[slice_along_axis(0, 1, dim=dim)], result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else elem[
                slice_along_axis(0, 1, dim=dim)
            ]  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(elems_flat)
    
    # TODO: With this, the requires_grad=True for the results
    # scans = elems_flat
    # TODO: With this, the requires_grad=False for the results
    # scans = operator(*elems_flat, *elems_flat)#, *())
    # scans = operator(*elems_flat, *elems_flat, *elems_flat, *elems_flat, *())

    return scans

def generic_associative_scan_old(operator, elems_flat, dim=0, spec='*', lifted_args=()):
    # TODO: The recursion involved here "unrolls" the scan
    # function for all inputs. Could there be a more efficient
    # way instead of running over the operation in sequence?
    def _scan(elems):
    
        """Perform scan on `elems`."""
        # num_elems = elems[0].shape[dim]
        num_elems = 3
        reduced_elems = elems
        scan_odd_part = []
        while num_elems > 2:
            # num_elems = reduced_elems[0].shape[dim]
            num_elems = 1

            # reduced_elems = operator(
            #     *[elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in reduced_elems],
            #     *[elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in reduced_elems],
            #     #*lifted_args,
            # )
            
            lhs = [elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in reduced_elems][0]
            # lhs = pytree.tree_unflatten([elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in reduced_elems], spec)
            rhs = [elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in reduced_elems][0]
            # rhs = pytree.tree_unflatten([elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in reduced_elems], spec)
            
            reduced_elems = operator(lhs, rhs)
            
            # reduced_elems = pytree.tree_leaves(reduced_elems)
            reduced_elems = [reduced_elems]
            
            scan_odd_part.append((num_elems, elems, reduced_elems))
            
            elems = reduced_elems
            
        return elems
        
        # # Recursively compute scan for partially reduced tensors.
        # odd_elems = _scan(reduced_elems)
        
        scan_odd_part.pop()
        num_elems, elems, odd_elems = scan_odd_part.pop()

        while len(scan_odd_part) > 0:
            
            if num_elems % 2 == 0:
                # even_elems = operator(
                #     *[e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems],
                #     *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                #     # *lifted_args,
                # )
                
                # lhs = [e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems][0]
                lhs = pytree.tree_unflatten([e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems], spec)
                # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
                rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems], spec)
            
                even_elems = operator(
                    lhs,
                    rhs,
                    # *lifted_args,
                )
                
                even_elems = pytree.tree_leaves(even_elems)
                # even_elems = [even_elems]
            
            else:
                # even_elems = operator(
                #     *odd_elems,
                #     *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                #     # *lifted_args,
                # )
                
                # lhs = odd_elems[0]
                lhs = pytree.tree_unflatten(odd_elems, spec)
                # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
                rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems], spec)

                even_elems = operator(
                    lhs,
                    rhs,
                    # *lifted_args,
                )
                
                even_elems = pytree.tree_leaves(even_elems)
                # even_elems = [even_elems]

            # The first element of a scan is the same as the first element
            # of the original `elems`.
            even_elems = [
                torch.cat([elem[slice_along_axis(0, 1, dim=dim)], result], dim=dim)
                if result.shape.numel() > 0 and elem.shape[dim] > 0
                else result
                if result.shape.numel() > 0
                else elem[
                    slice_along_axis(0, 1, dim=dim)
                ]  # Jax allows/ignores concat with 0-dim, Pytorch does not
                for (elem, result) in zip(elems, even_elems)
            ]

            odd_elems = list(
                safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
            )
            
            num_elems, elems, _ = scan_odd_part.pop()
            
        if num_elems % 2 == 0:
            # even_elems = operator(
            #     *[e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems],
            #     *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
            #     # *lifted_args,
            # )
            
            # lhs = [e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems][0]
            lhs = pytree.tree_unflatten([e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems], spec)
            # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
            rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems], spec)
        
            even_elems = operator(
                lhs,
                rhs,
                # *lifted_args,
            )
            
            even_elems = pytree.tree_leaves(even_elems)
            # even_elems = [even_elems]
        
        else:
            # even_elems = operator(
            #     *odd_elems,
            #     *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
            #     # *lifted_args,
            # )
            
            # lhs = odd_elems[0]
            lhs = pytree.tree_unflatten(odd_elems, spec)
            # rhs = [e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems][0]
            rhs = pytree.tree_unflatten([e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems], spec)

            even_elems = operator(
                lhs,
                rhs,
                # *lifted_args,
            )
            
            even_elems = pytree.tree_leaves(even_elems)
            # even_elems = [even_elems]

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([elem[slice_along_axis(0, 1, dim=dim)], result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else elem[
                slice_along_axis(0, 1, dim=dim)
            ]  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]
            
        return list(
                safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
            )

    scans = _scan(elems_flat)
    
    # TODO: With this, the requires_grad=True for the results
    # scans = elems_flat
    # TODO: With this, the requires_grad=False for the results
    # scans = operator(*elems_flat, *elems_flat)#, *())
    # scans = operator(*elems_flat, *elems_flat, *elems_flat, *elems_flat, *())

    return scans


associative_scan_op = HigherOrderOperator("associative_scan")


def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    input: List[torch.Tensor],
    dim: int,
    lifted_args: Tuple[torch.Tensor],
):
    from torch.fx.experimental.proxy_tensor import maybe_handle_decomp

    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.empty_like(
                x[slice_along_axis(0, 1, stride=None, dim=dim)],
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in itertools.chain(input, input)
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs, *lifted_args)
        
    # sample_inputs = [
    #     x[slice_along_axis(0, 1, stride=None, dim=dim)]
    #     for x in itertools.chain(input, input)
    # ]
    # sample_inputs = [
    #     torch.empty_like(
    #         x,
    #         # x[slice_along_axis(0, 1, stride=None, dim=dim)],
    #         dtype=x.dtype,
    #         device=x.device,
    #         requires_grad=x.requires_grad,
    #     )
    #     for x in itertools.chain(input, input)
    # ]
    # combine_graph = reenter_make_fx(combine_fn)(*sample_inputs, *lifted_args)

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

    for i, o in zip(input, outputs):
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, input, dim, lifted_args)
    out = maybe_handle_decomp(proxy_mode, associative_scan_op, args, {})
    if out is not NotImplemented:
        return out

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in input]

    # # TODO: the unbacked symbol allocations MUST NOT leak out, if you want to
    # # support this we need to arrange for the reenter_make_fx unbacked SymInts
    # # to be used, AND we need to arrange for some sort of unification between
    # # the two branches (but not really unification; e.g., if one branch
    # # returns [u0] and the other returns [5] this is OK but you MUST NOT
    # # conclude the result is 5.  Also if one branch returns [3] and another
    # # branch returns [5] you can make it work by immediately allocating a new
    # # unbacked SymInt here).
    # ignore_fresh_unbacked = contextlib.nullcontext()
    # if (fake_mode := detect_fake_mode()) and fake_mode.shape_env:
    #     ignore_fresh_unbacked = fake_mode.shape_env.ignore_fresh_unbacked_symbols()

    # with ignore_fresh_unbacked:
    #     out = combine_fn(*sample_inputs, *lifted_args)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)

class AssociativeScanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fw_true_graph,
        joint_true_graph,
        dim,
        spec,
        *input,
        # lifted_args,
    ):
        ctx._dim = dim
        ctx._fw_true_graph = fw_true_graph
        ctx._joint_true_graph = joint_true_graph
        # ctx._lifted_args = lifted_args
        ctx.save_for_backward(*input)

        with torch._C._AutoDispatchBelowAutograd():
            # return associative_scan_op(fw_true_graph, input, dim, spec, ())#, lifted_args)
            outs = associative_scan_op(fw_true_graph, input, dim, spec, ())#, lifted_args)
            
            num_elems = input[0].size()[dim]
            
            helpers = [joint_true_graph(torch.ones_like(input[0][slice_along_axis(0, 1, dim=dim)]), x, z) for x, z in zip(torch.flip(input[0], [dim])[:-1], torch.flip(outs[0], [dim])[1:])]
            helper1 = [[h[0] for h in h_tree] for h_tree in helpers]
            helper1.append(torch.ones_like(input[0][slice_along_axis(0, 1, dim=dim)]))
            helper1 = torch.concat(helper1, dim)
            helper2 = [torch.ones_like(input[0][slice_along_axis(0, 1, dim=dim)])]
            helper2.extend([[h[1] for h in h_tree] for h_tree in helpers])
            helper2 = torch.concat(helper2, dim)
            
            helper_mat = []
            for n in range(num_elems):
                row = [torch.zeros_like(input[0][slice_along_axis(0, 1, dim=dim)])] * n + [torch.ones_like(input[0][slice_along_axis(0, 1, dim=dim)])]
                row.append(torch.cumprod(helper2[slice_along_axis(n+1, None, dim=dim)], dim))
                row = torch.concat(row, dim)
                helper_mat.append(row)
                
            helper_mat = torch.stack(helper_mat, 0)
            
            grads = torch.flip(torch.sum(helper1 * helper_mat, 0), [dim])
            print(grads)
            
            return outs, grads

    @staticmethod
    def backward(ctx, *flat_grads):
        input = ctx.saved_tensors

        # TODO: Compute gradients
        return None, None, None, None, None, *grads

@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, input, dim, spec, lifted_args):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_associative_scan(combine_fn, input, dim, spec, lifted_args)


@associative_scan_op.py_impl(DispatchKey.Autograd)
def associative_scan_op_autograd(combine_fn, input, dim, spec, lifted_args):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,
        (input),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return associative_scan_op(combine_fn, input, dim, lifted_args)
        
    # combine_fn = functools.partial(
    #     wrap_combine_fn_flat2, combine_fn=combine_fn.keywords['combine_fn'], spec=combine_fn.keywords['spec'], num_leaves=combine_fn.keywords['num_leaves']
    #     # wrap_combine_fn_flat, combine_fn=combine_fn.keywords['combine_fn'], spec=combine_fn.keywords['spec'], num_leaves=combine_fn.keywords['num_leaves']
    # )
        
    (
        fw_true_graph,
        joint_true_graph,
    # TODO: When creating the graph like this, there are NO gradients
    ) = create_fw_bw_graph_combinefn(combine_fn, dim, *input)#lifted_args)
    # TODO: When creating the graph like this, there are gradients
    # ) = create_fw_bw_graph_combinefn(combine_fn.keywords['combine_fn'], dim, *input)#lifted_args)
    
    # (
    #     fw_true_wrapped_graph
    # ) = create_fw_bw_graph_wrapfn(wrap_combine_fn_flat2, fw_true_graph, *input)#lifted_args)
    
    
    # # TODO: Wrapping the function after creating the graph does also not help
    # wrapped = functools.partial(
    #     fw_true_wrapped_graph, combine_fn=fw_true_graph, spec=combine_fn.keywords['spec'], num_leaves=combine_fn.keywords['num_leaves']
    # )
    
    # (
    #     fw_partial_graph
    # ) = create_fw_bw_graph_partial(wrapped, *input)#lifted_args)
    
    
    flat_out, grads = AssociativeScanAutograd.apply(
        # fw_partial_graph,
        # fw_true_wrapped_graph,
        fw_true_graph,
        joint_true_graph,
        dim,
        spec,
        *input,
        # lifted_args
    )
    return flat_out, grads


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, input, dim, spec, lifted_args):
    if mode.enable_tracing:
        return trace_associative_scan(
            mode, associative_scan_op, combine_fn, input, dim, spec, lifted_args
        )
    else:
        return associative_scan_op(
            mode, associative_scan_op, combine_fn, input, dim, spec, lifted_args
        )


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, spec, dim, lifted_args):
    with mode:
        return [x.clone() for x in input]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, input, dim, spec, lifted_args):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_lifted_args = ctx.unwrap_tensors(lifted_args)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        ret = associative_scan_op(
            functional_combine_fn, unwrapped_input, dim, spec, unwrapped_lifted_args
        )
    return ctx.wrap_tensors(ret)


@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, input, dim, spec, combine_fn):
    input_ = [get_unwrapped(x) for x in input]
    input_bdims = [maybe_get_bdim(x) for x in input]

    batch_size = None
    for inp, bdim in zip(input, input_bdims):
        if bdim is not None:
            batch_size = get_unwrapped(inp).shape[bdim]

    assert batch_size
    input_unwrapped = []
    for x, bdim in zip(input, input_bdims):
        unwrap = get_unwrapped(x)
        if dim is None:
            unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
        else:
            unwrap = unwrap.movedim(bdim, 0)
        input_unwrapped.append(unwrap)

    res = associative_scan_op(combine_fn, input_unwrapped, dim + 1)
    lvl = interpreter.level()
    return [_add_batch_dim(x, 0, lvl) for x in res]
