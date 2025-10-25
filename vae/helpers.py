import os, platform
from collections.abc import Iterator
from typing import Generic, TypeVar, overload
from torch import nn, Tensor
from urllib.request import urlopen
from pathlib import Path
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

# misc features
def dl_cache(url:str, fn:str) -> str:
  if platform.system() == "Darwin": cache = Path.home() / "Library" / "Caches" / "vae"
  else: cache = Path.home() / ".cache" / "vae"
  cache.mkdir(parents=True, exist_ok=True)
  if not (save_path := cache / fn).exists():
    with urlopen(url) as r, open(save_path, "wb") as f: f.write(r.read())
  return save_path

# from tinygrad
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))