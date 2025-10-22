import os
from collections.abc import Iterator
from typing import Generic, TypeVar, overload
from torch import nn, Tensor
T = TypeVar("T", bound=nn.Module)

# from tinygrad
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

# from https://github.com/pytorch/pytorch/issues/80821
class ModuleListTyped(Generic[T], nn.ModuleList):
  def __iter__(self) -> Iterator[T]:
    return super().__iter__()  # type: ignore[no-any-return]

  def append(self, module: T) -> "ModuleListTyped[T]":  # type: ignore[override]
    return super().append(module)  # type: ignore[return-value]

  @overload
  def __getitem__(self, idx: slice) -> "ModuleListTyped[T]": ...

  @overload
  def __getitem__(self, idx: int) -> T: ...

  def __getitem__(self, idx):  # type: ignore[no-untyped-def]
    return super().__getitem__(idx)

  def __setitem__(self, idx: int, module: T) -> None:  # type: ignore[override]
    super().__setitem__(idx, module)

# based on https://discuss.pytorch.org/t/adding-typing-to-call-of-nn-module/118295
class ModuleCallTyped(nn.Module):
  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)

class Conv2dTyped(ModuleCallTyped, nn.Conv2d): pass

class GroupNormTyped(ModuleCallTyped, nn.GroupNorm): pass