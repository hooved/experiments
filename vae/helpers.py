import os
from collections.abc import Iterator
from typing import Generic, TypeVar, overload
from torch import nn, Tensor
T = TypeVar("T", bound=nn.Module)

# These helpers enable correct static type checking

def register_buffer(module:nn.Module, name:str, buf:Tensor) -> Tensor:
  module.register_buffer(name, buf)
  ret = getattr(module, name)
  assert isinstance(ret, Tensor)
  return ret
  
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
class ModuleTyped(nn.Module):
  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)

class Conv2dTyped(ModuleTyped, nn.Conv2d): pass

class GroupNormTyped(ModuleTyped, nn.GroupNorm): pass

class SequentialTyped(ModuleTyped, nn.Sequential):
  @overload
  def __getitem__(self, idx: slice) -> "SequentialTyped": ...

# from tinygrad
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))