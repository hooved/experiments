from collections.abc import Iterator
from typing import Generic, TypeVar, overload
from torch import nn
T = TypeVar("T", bound=nn.Module)

# from https://github.com/pytorch/pytorch/issues/80821
class TypedModuleList(Generic[T], nn.ModuleList):
    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore[no-any-return]

    def append(self, module: T) -> "TypedModuleList[T]":  # type: ignore[override]
        return super().append(module)  # type: ignore[return-value]

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[T]": ...

    @overload
    def __getitem__(self, idx: int) -> T: ...

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: T) -> None:  # type: ignore[override]
        super().__setitem__(idx, module)