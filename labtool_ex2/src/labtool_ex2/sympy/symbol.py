from sympy import Symbol as SimpSymbol

from typing import Any, Iterator
import numpy as np
from pandas._typing import PositionalIndexer, AnyArrayLike
from pandas import DataFrame
import labtool_ex2 as lab


# I need a symbol which is backed by data and allows easy access
class Symbol(SimpSymbol):
    # def __new__(cls, name, df: DataFrame = DataFrame(), **assumptions):
    #     super.__new__(name, **assumptions)
    def __new__(cls, name, project: lab.Project = lab.Project(None), **assumptions):
        # obj = SimpSymbol.__new__(cls, name, **assumptions)
        # print(f"Hello {assumptions}")
        # print(f"world {df}")
        # try:
        #     del assumptions["df"]
        # except KeyError:
        #     pass
        # print(df)

        obj = super().__new__(cls, name, **assumptions)
        # print(f"bruh {dir(obj)}")
        obj._project: lab.Project = project
        # obj._df["bruh"] = [0, 4]
        return obj

    # def __new__(cls, name, df: DataFrame = DataFrame(), **assumptions):
    #     # obj = SimpSymbol.__new__(cls, name, **assumptions)
    #     # print(f"Hello {assumptions}")
    #     # print(f"world {df}")
    #     # try:
    #     #     del assumptions["df"]
    #     # except KeyError:
    #     #     pass
    #     # print(df)

    #     obj = super().__new__(cls, name, **assumptions)
    #     # print(f"bruh {dir(obj)}")
    #     obj._df = df
    #     obj._df["bruh"] = [0, 4]
    #     return obj

    # def __init__(self, name, df):
    #     # print(df)
    #     # self._df: DataFrame
    #     # super().__init__(name)
    #     super().__init__()

    # lazy evaluation
    @property
    def data(self) -> AnyArrayLike:
        # try:
        return self._project.data[self.name]

    # @property
    # def data(self) -> AnyArrayLike:
    #     # try:
    #     return self._df[self.name]
    # except KeyError:
    # pass
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
