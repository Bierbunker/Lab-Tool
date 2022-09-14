from __future__ import annotations
from typing import Callable, Union, Any, Sequence, Optional
from sympy import Symbol as SimpSymbol

from typing import Any, Iterator
import numpy as np
from pandas._typing import PositionalIndexer, AnyArrayLike
from pandas import DataFrame
from sympy import Basic, FunctionClass
from sympy import symbols

from sympy.utilities.iterables import NotIterable

# from .project import Project

# from . import project as p
import labtool_ex2.project as p


# I need a symbol which is backed by data and allows easy access
class Symbol(SimpSymbol, NotIterable):
    def __new__(cls, name, project: Optional[p.Project] = None, **assumptions):
        if project is None:
            project = p.Project(None)

        obj = super().__new__(cls, name, **assumptions)
        obj._project = project  # type: ignore
        return obj

    def __init__(self, name, project: Optional[p.Project] = None):
        if project is None:
            project = p.Project(None)
        self._project: p.Project
        super().__init__()

    # lazy evaluation
    @property
    def data(self) -> AnyArrayLike:
        # try:
        return self._project.data[self.name]

    # def __len__(self) -> int:
    #     return len(self.data)

    # @property
    # def data(self) -> AnyArrayLike:
    #     # try:
    #     return self._df[self.name]
    # except KeyError:
    # pass
    # This Symbol isn't backed by any data like the outlandish claims of Donald J. Trump
    # raise KeyError(
    #     f"Couldn't find '{self.name}' as an index in the Data-Source try loading some values!"
    # )

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self.data)):
            yield self.data[i]

    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        if item in self.data:
            return True
        else:
            return False

    def __getitem__(self, item: PositionalIndexer) -> Any:
        return self.data[item]


def _custom_var(names: list[str], project: p.Project, **args):  # noqa
    """
    Create symbols and inject them into the global namespace.

    Explanation
    ===========

    This calls :func:`symbols` with the same arguments and puts the results
    into the *global* namespace. It's recommended not to use :func:`var` in
    library code, where :func:`symbols` has to be used::

    Examples
    ========

    >>> from sympy import var

    >>> var(['x'])
    x
    >>> x # noqa: F821
    x

    >>> var(["a","ab","abc"])
    (a, ab, abc)
    >>> abc # noqa: F821
    abc

    >>> var(["x","y"], real=True)
    (x, y)
    >>> x.is_real and y.is_real # noqa: F821
    True

    See :func:`symbols` documentation for more details on what kinds of
    arguments can be passed to :func:`var`.

    """

    # print(names)

    def traverse(symbols, frame):
        """Recursively inject symbols to the global namespace."""
        for symbol in symbols:
            if isinstance(symbol, Basic):
                frame.f_globals[symbol.name] = symbol  # type: ignore
                # magic see https://stackoverflow.com/questions/34650744/modify-existing-variable-in-locals-or-frame-f-locals
                # if symbol.name in frame.f_locals:  # type: ignore
                import ctypes

                frame.f_locals.update({symbol.name: symbol})  # type: ignore
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
            elif isinstance(symbol, FunctionClass):
                frame.f_globals[symbol.__name__] = symbol
                # if symbol.__name__ in frame.f_locals:  # type: ignore
                import ctypes

                frame.f_locals.update({symbol.__name__: symbol})  # type: ignore
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
            else:
                traverse(symbol, frame)

    from inspect import currentframe

    frame = currentframe().f_back.f_back  # type: ignore
    # print(frame)
    # print(frame.f_globals)
    # print(frame.f_locals)

    try:
        syms = symbols(names, cls=Symbol, project=project, **args)

        if syms is not None:
            if isinstance(syms, Basic):
                frame.f_globals[syms.name] = syms  # type: ignore
                # if syms.name in frame.f_locals:  # type: ignore
                import ctypes

                frame.f_locals.update({syms.name: syms})  # type: ignore
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
            elif isinstance(syms, FunctionClass):
                frame.f_globals[syms.__name__] = syms  # type: ignore
                # if syms.__name__ in frame.f_locals:  # type: ignore
                import ctypes

                frame.f_locals.update({syms.__name__: syms})  # type: ignore
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
            else:
                traverse(syms, frame)
    finally:
        # print(frame)
        # print(frame.f_globals)
        # print(frame.f_locals)
        del frame  # break cyclic dependencies as stated in inspect docs

    return syms
