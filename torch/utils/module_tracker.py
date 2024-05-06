import weakref

from typing import Set

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


class ModuleTracker:
    """
    ``ModuleTracker`` is a context manager that tracks the nn.Module hierarchy during execution
    so that other system can query which Module is currently being executed (or its backward is being
    executed).

    You can access the ``parents`` attribute on this context manager to get the set of all the
    Modules currently being executed via their fqn (fully qualified name, also used as the key within
    the state_dict).
    You can access the ``is_bw`` attribute to know if you are currently running in backward or not.

    Note that ``parents`` is never empty and always contains the "Global" key. The ``is_bw`` flag
    will remain ``True`` after the forward until another Module is executed. If you need it to be
    more accurate, please submit an issue requesting this. Adding a map from fqn to the module instance
    is possible but not done yet, please submit an issue requesting this if you need it.

    Example usage

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias
            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    parents: Set[str]
    """
    A Set containing the fqn for each module currently running their forward
    """

    is_bw: bool
    """
    A boolean marking if this is currently running during the backward pass or not
    """

    def __init__(self):
        self.parents = {"Global"}
        # This is used to reset parents at the end of the backward
        self.is_bw = False
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules = set()

    def _get_mod_name(self, mod):
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
        return mod_name

    def _get_append_fn(self, name, is_bw):
        def fn(*args):
            if self.is_bw != is_bw:
                self.parents = {"Global"}
                self.is_bw = is_bw
            if name in self.parents:
                print(
                    "The module hierarchy tracking seems to be messed up."
                    "Please file a bug to PyTorch."
                )
            self.parents.add(name)

        return fn

    def _get_pop_fn(self, name, is_bw):
        def fn(*args):
            if name in self.parents:
                self.parents.remove(name)
            elif not is_bw:
                # Due to some input/output not requiring gradients, we cannot enforce
                # proper nesting in backward
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    def _fw_pre_hook(self, mod, input):
        name = self._get_mod_name(mod)
        self._get_append_fn(name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, self._get_pop_fn(name, True))

    def _fw_post_hook(self, mod, input, output):
        name = self._get_mod_name(mod)
        self._get_pop_fn(name, False)()

        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, self._get_append_fn(name, True))

    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        return self

    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
