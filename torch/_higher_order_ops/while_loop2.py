import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint, from_fun

from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.multiprocessing.reductions import StorageWeakRef


# TODO: We add this to prevent dymamo from tracing into map_wrapper,
# remove the wrapper call when it's ready.
class WhileLoopWrapper(HigherOrderOperator):
    def __call__(self, cond_fn, body_fn, operands):
        return while_loop_wrapper(cond_fn, body_fn, operands)


while_loop = WhileLoopWrapper("while_loop")
while_loop_impl = HigherOrderOperator("while_loop_impl")

dummy_aot_config = AOTConfig(
    fw_compiler=None,  # type: ignore[arg-type]
    bw_compiler=None,  # type: ignore[arg-type]
    partition_fn=None,  # type: ignore[arg-type]
    decompositions={},
    num_params_buffers=0,
    aot_id=0,
    keep_inference_input_mutations=False,
)


def create_fw_bw_graph(body_fn, num_operands, operands):

    # Note: We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            def _from_fun(t):
                if isinstance(t, torch.Tensor):
                    if t.dtype != torch.bool:
                        return torch.empty_strided(
                            t.size(),
                            t.stride(),
                            dtype=t.dtype,
                            requires_grad=t.requires_grad,
                        )
                    else:
                        # clone of a functional tensor produces a functional tensor
                        # but we want to avoid it so we clone a non-functional version
                        maybe_unfunc_t = t
                        if isinstance(t, FunctionalTensor):
                            torch._sync(t)
                            maybe_unfunc_t = from_fun(t)
                        elif torch._is_functional_tensor(t):
                            # need to handle both types of functionalization here:
                            # these are the tensors that came from the user,
                            # which could be either FunctionalTensorWrapper or FunctionalTensor
                            torch._sync(t)
                            maybe_unfunc_t = torch._from_functional_tensor(t)
                        return maybe_unfunc_t.clone()
                return t

            unwrapped_mapped_operands = [pytree.tree_map(_from_fun, op) for op in operands]
            example_operands = tuple(unwrapped_mapped_operands)
            example_flat_out = pytree.tree_map(
                    _from_fun, body_fn(*example_operands)
                )
            
            if any(
                not isinstance(out, torch.Tensor)
                for out in example_operands
                if out is not None
            ):
                raise RuntimeError(
                    "Expect operands of while to only contain tensors or None. "
                    f"Got types {[type(out) for out in example_flat_out]}."
                )
            example_grad = tuple([_from_fun(out) for out in example_flat_out])

            fw_graph = make_fx(body_fn)(*example_operands)#[:num_operands])

        def joint_f(*example_args):
            mapped_input = example_args[:num_operands]
            mapped_outputs = example_args[num_operands:2*num_operands]
            mapped_grads = example_args[2*num_operands:3*num_operands]

            def fw_with_masks(*args):
                fw_out = body_fn(*args[:num_operands])
                grads = [
                    True
                    if isinstance(ret, torch.Tensor) and ret.requires_grad
                    else False
                    for ret in fw_out
                ]
                return fw_out, grads

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            _, grads = joint(
                # list(mapped_input) + list(mapped_outputs) + list(mapped_grads),
                list(mapped_input) + list(mapped_grads),
                [
                    grad
                    for grad in mapped_grads
                    if grad is not None and grad.requires_grad
                ],
            )

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            input_storage = {
                StorageWeakRef(arg._typed_storage())
                for arg in example_args
                if isinstance(arg, torch.Tensor)
            }

            def maybe_clone(t):
                if (
                    isinstance(t, torch.Tensor)
                    and StorageWeakRef(t._typed_storage()) in input_storage
                ):
                    return t.clone()
                return t

            return pytree.tree_map(maybe_clone, grads)

        joint_graph = make_fx(joint_f)(*example_operands, *example_flat_out, *example_grad)

        return fw_graph, joint_graph


def while_loop_wrapper(cond_fn, body_fn, operands):
    flat_operand, operand_spec = pytree.tree_flatten(operands)
    num_operands = len(flat_operand)

    #TODO: Introduce checks on shapes and properties of operands
    # if not all(isinstance(t, torch.Tensor) for t in flat_xs):
    #     raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    # num_mapped_args = len(flat_xs)
    # shapes = [xs.shape for xs in flat_xs]
    # leading_dim_size = shapes[0][0]
    # if leading_dim_size == 0:
    #     raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    # if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
    #     raise RuntimeError(
    #         f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
    #     )

    out_spec = None

    def flat_fn(*flat_args):
        xs = pytree.tree_unflatten(list(flat_args), operand_spec)
        unflattened_out = body_fn(*xs)
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        nonlocal out_spec
        out_spec = tmp_out_spec
        return flat_out

    return pytree.tree_unflatten(
        while_loop_impl(cond_fn, flat_fn, flat_operand), out_spec  # type: ignore[arg-type]
    )


class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond_fn, fw_graph, joint_graph, num_operands, *flat_operands):
        #ctx.save_for_backward(*flat_operands)
        ctx._fwd_operands = flat_operands
        ctx._num_operands = num_operands
        ctx._cond_fn = cond_fn
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        with torch._C._AutoDispatchBelowAutograd():
            #ctx.save_for_backward(*flat_operands)
            #outs = while_loop_impl(cond_fn, fw_graph, [flat_operands, flat_operands])
            #outs = while_loop_impl(cond_fn, fw_graph, ctx._num_operands, flat_operands)
            outs = while_loop_impl(cond_fn, fw_graph, flat_operands)
            #print(outs)
            for o in outs:
                #ctx.save_for_backward(flat_operands[0])
                ctx.save_for_backward(o)
            return tuple([o[-1:] for o in outs])
            # return (
            #     *while_loop_impl(cond_fn, fw_graph, flat_operands),
            # )

    @staticmethod
    def backward(ctx, *flat_grads):
        # vals = ctx.saved_tensors
        # fw_operands = vals[:ctx._num_operands]
        fw_operands = ctx._fwd_operands
        fw_outs = ctx.saved_tensors#[ctx._num_operands:2*ctx._num_operands]
        # fw_mapped_args = fw_args[: ctx._num_mapped_args]
        # pos_args = fw_args[ctx._num_mapped_args :]

        # grads = while_loop_impl(
        #     ctx._cond_fn,
        #     ctx._joint_graph,
        #     [fw_outs,
        #     flat_grads]
        # )
        #grads = while_loop_impl(ctx._cond_fn, ctx._fw_graph, [fw_outs, flat_grads])
        grads = while_loop_impl(ctx._cond_fn, 
                                #ctx._fw_graph,
                                ctx._joint_graph,
                                #ctx._num_operands,
                                fw_operands + fw_outs + flat_grads,
                                #fw_operands
                                #fw_outs + flat_grads
                                #fw_outs
                                )
        return None, None, None, *grads


def trace_while_loop(proxy_mode, while_loop_op, cond_fn, body_fn, operands):

    pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)
    with disable_proxy_modes_tracing():
        cond_graph = reenter_make_fx(cond_fn, pre_dispatch)(*operands)
        body_graph = reenter_make_fx(body_fn, pre_dispatch)(*operands)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"while_loop_cond_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate
    cond_graph_name = next_name
    body_graph_name = f"while_loop_body_graph_{i}"
    assert not hasattr(proxy_mode.tracer.root, body_graph_name)

    proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    args = (cond_graph, body_graph, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", while_loop_op, proxy_args, {}, name="while_loop"
    )

    # body_fn return output with the same pytree and tensor meta data as operands
    # so we could just return the output after one iteration.
    out = body_fn(*operands)
    return track_tensor_tree(
        out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    a = zip(*flat_xs)

    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees


def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    assert out_spec is not None
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


@while_loop_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, operands):
    #Determine whether the FW or the BW pass is being computed
    num_operands = len(operands)
    if len(operands) > num_operands:
        fw_mode = False
        fw_operands = operands[:num_operands]
        init_val = fw_operands
        fw_outs = operands[num_operands:2*num_operands]
        grads = operands[2*num_operands:3*num_operands]
        #For the last time step, use the last output
        g = tuple([fwo[-1] for fwo in fw_outs])
        ind = 0
    else:
        init_val = operands
        fw_mode = True

    def _is_boolean_scalar_tensor(pred):
        return (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        )

    if not isinstance(init_val, tuple):
        raise RuntimeError(f"operands must be a tuple but got {type(init_val)}")

    if fw_mode:
        outs = [init_val]
    else:
        outs = [grads]
    while pred := cond_fn(*init_val):
        if not _is_boolean_scalar_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        if fw_mode:
            out = tuple(body_fn(*init_val))
            outs.append(out)
        else:
            #Call the backward function of the body
            out_vals = [fwo[ind] for fwo in fw_operands]
            grads = tuple(body_fn(num_operands, *out_vals, *grads))
            # grads *= g
            outs.append(grads)
            ind += 1
        assert isinstance(
            out, tuple
        ), f"body_fn should return a tuple but got {type(out)}"
        assert len(out) == len(
            init_val
        ), "body_fn should return the same number of elements as operands"
        init_val = out
    #return init_val
    stacked_val = list(outs[0])
    for vals_t in outs[1:]:
        # st_tmp = []
        # for vals in outs:
        #     st_tmp.append(vals[vals_inds])
        for ind, vals in enumerate(vals_t):
            stacked_val[ind] = torch.cat([stacked_val[ind], vals])
    return stacked_val


@while_loop_impl.py_impl(DispatchKey.Autograd)
def while_loop_autograd(cond_fn, body_fn, operands):
    # dummy_inps = tuple([torch.ones(op.shape, requires_grad=True) for op in operands])
    # if len(operands) > num_operands:
    #     dummy_outs = body_fn_grad(num_operands, *(dummy_inps[0:num_operands] + dummy_inps[-num_operands:]))
    # else:
    # dummy_outs = body_fn(*dummy_inps)
    # grad_fn = [op.grad_fn if (hasattr(op, 'grad_fn') and op.grad_fn is not None) else None for op in dummy_outs]
    # def body_fn_grad(num_operands, *args):
    #     # fw_operands = args[:num_operands]
    #     # fw_outs = args[num_operands:2*num_operands]
    #     # grads = args[2*num_operands:3*num_operands]
        
    #     fw_outs = args[:num_operands]
    #     grads = args[num_operands:2*num_operands]
        
    #     #grads = [fn(arg) for fn, arg in zip(grad_fn, list(args))]
    #     grads = [fn(arg)[0] * g if fn is not None else None for fn, arg, g in zip(grad_fn, fw_outs, grads)]
    #     #return tuple(grads[:num_operands])
    #     return tuple(grads)
    
    num_operands = len(operands)
    fw_graph, bw_graph = create_fw_bw_graph(body_fn, num_operands, operands)
    flat_out = WhileLoopAutogradOp.apply(cond_fn, fw_graph, bw_graph, num_operands, *operands)
    return flat_out


@while_loop_impl.py_impl(ProxyTorchDispatchMode)
def while_loop_proxy_torch_dispatch_mode(mode, cond_fn, body_fn, operands):
    if mode.enable_tracing:
        return trace_while_loop(mode, while_loop_impl, cond_fn, body_fn, operands)
    else:
        return while_loop_impl(cond_fn, body_fn, operands)


@while_loop_impl.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(mode, cond_fn, body_fn, operands):
    return body_fn(*operands)


@while_loop_impl.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    num_operands = len(unwrapped_operands)
    with ctx.redispatch_to_next() as m:
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        for fn, fn_name in [
            (functional_cond_fn, "cond_fn"),
            (functional_body_fn, "body_fn"),
        ]:
            if _has_potential_branch_input_mutation(fn, unwrapped_operands):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

        for fn in [functional_cond_fn, functional_body_fn]:
            if _has_potential_branch_input_alias(fn, unwrapped_operands):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        ret = while_loop_impl(functional_cond_fn, functional_body_fn, unwrapped_operands)
        return ctx.wrap_tensors(ret)