import numpy as np
import pandas as pd
from uncertainties import unumpy
from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    register_extension_dtype,
    register_series_accessor,
    register_dataframe_accessor,
)
from pandas._typing import (
    DtypeObj,
    Dtype,
    PositionalIndexer,
)
from pandas.api.types import is_complex_dtype
from typing import TypeVar, Any
from pandas.core.arrays.base import ExtensionOpsMixin
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc, Variable
from uncertainties.unumpy import uarray
from uncertainties import ufloat_fromstr

from collections.abc import Iterable
from pandas.api.types import is_integer, is_list_like
from pandas.core.indexers import check_array_indexer

# own project imports
from .uarray import UArray

UfloatDtypeT = TypeVar("UfloatDtypeT", bound="UfloatDtype")


@register_extension_dtype
class UfloatDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray"""

    type: type = Variable
    # linter wants a property, but pandas validates the type of name
    # as a string via assertion, therefore: AssertionError if written
    # as a property
    name: str = "ufloat"

    @property
    def _is_numeric(self):
        return True

    @property
    def _is_boolean(self):
        return True

    @property
    def na_value(self):
        return ufloat(np.nan, np.nan)

    _common_dtypes: list[DtypeObj] = [
        np.dtype(np.single),
        np.dtype(np.double),
        np.dtype(object),
    ]
    # _metadata: tuple[str, ...] = (
    #     "value",
    #     "std_dev",
    # )
    _match: str = r"(\d((\.|,)\d)?)\+\\\-(\d((\.|,)\d)?)"

    def __new__(cls, unit=None):
        """
        Creation is allowed for :
          - already UfloatDtype isinstance
          - a string like according to uncertainties
          - None
          - a quantity
        """
        if isinstance(unit, str):
            unit = cls._parse_dtype_strict(unit)
        elif unit is None:
            unit = ufloat(np.nan, np.nan)

        # Now that we have a quantity, we create the Dtype
        if isinstance(unit, AffineScalarFunc):
            # qdtype_unit = QuantityDtype(unit)
            u = object.__new__(cls)
            return u
        else:
            raise ValueError

    @classmethod
    def construct_array_type(cls, *args):
        """
        Return the array type associated with this dtype
        Return UfloatArray
        -------
        type
        """
        return UfloatArray

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            for dtype in dtypes:
                if dtype in self._common_dtypes:
                    return dtype
            return None

    def __repr__(self):
        """
        Return a string representation for this object.
        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """
        return self.name

    @classmethod
    def _parse_dtype_strict(cls, string_unit):
        """
        Parses the ufloat, which should be a string like
        """
        if isinstance(string_unit, str) and string_unit == "ufloat":
            return ufloat_fromstr(string_unit)

        raise ValueError(f"could not construct UfloatDtype with {string_unit}.")


UfloatArrayT = TypeVar("UfloatArrayT", bound="UfloatArray")


class UfloatArray(ExtensionArray, ExtensionOpsMixin):
    """
    The interface includes the following abstract methods that must be
    implemented by subclasses:

    * _from_sequence
    * _from_factorized
    * __getitem__
    * __len__
    * __eq__
    * dtype
    * nbytes
    * isna
    * take
    * copy
    * _concat_same_type
    """

    def __init__(self, variables, dtype: Dtype | None = None, copy: bool = False):
        # if (
        #     (isinstance(variables, list) or isinstance(variables, np.ndarray))
        #     and len(variables) == 1
        #     and isinstance(variables[0], AffineScalarFunc)
        # ):
        #     variables = variables
        self._data = UArray(variables)

        self._dtype = UfloatDtype()

    @property
    def dtype(self) -> UfloatDtype:
        """An instance of 'UfloatDtype'."""
        return self._dtype

    def __len__(self) -> int:
        """Length of this array."""
        return len(self._data)

    @classmethod
    def _from_sequence(cls, scalars, dtype: Dtype | None = None, copy: bool = False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        scalars = UArray(scalars)
        # print(f"hello {scalars}")
        # print(type(scalars[0]))
        # dtype = UfloatDtype()
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy=False, **kwargs
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: Iterable, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values, dtype=original.dtype)

    def astype(self, dtype: Dtype | str | AffineScalarFunc, copy: bool = True):
        """Cast to a NumPy array with 'dtype'.
        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.
        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        # print("HI")
        # assert False
        # return UArray(self._data)
        if isinstance(dtype, str) and (dtype == "ufloat"):
            dtype = UfloatDtype()
        if isinstance(dtype, UfloatDtype):
            if dtype == self._dtype and not copy:
                return self
            else:
                return self.copy()
        elif isinstance(dtype, AffineScalarFunc):
            return UfloatArray(self._data, UfloatDtype())
        # return self.__array__(dtype, copy)
        # return Variable(Quantity(self.quantity.value.astype(dtype), self.quantity.dimension)
        return UArray(self._data)
        # return np.array(self, dtype=dtype, copy=copy)

    def __getitem__(self, item: PositionalIndexer) -> UfloatArrayT | Any:
        """Select a subset of self."""
        # return self._data[item]
        if is_integer(item):
            # we use the __getitem__ of the underlying Quantity
            # to return a QuantityType.type == Quantity instance scalar
            return self._data[item]

        # seems to be a good practice
        item = check_array_indexer(self, item)

        # return another QuantityArray with the subset
        # again using __getitem__ of the Quantity
        return self.__class__(self._data[item], self._dtype)

    def __setitem__(self, key, value):
        """
        Necessary for some funtions.
        """
        if isinstance(value, AffineScalarFunc):
            value = value
        elif isinstance(value, str):
            value = ufloat_fromstr(value)
        elif (isinstance(value, tuple) or isinstance(value, list)) and (
            list(map(type, value)) == [int, int]
            or list(map(type, value)) == [float, float]
        ):
            value = ufloat(value[0], value[1])

        self._data[key] = value

    def round(self, decimals: int = 0, *args, **kwargs):
        """
        Used by round.
        """
        return type(self)(np.around(self._data, decimals), self._dtype)

    @property
    def _itemsize(self) -> int:
        from sys import getsizeof as sizeof

        return sizeof(Variable)

    def __eq__(self, other):
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        if (
            isinstance(other, pd.DataFrame)
            or isinstance(other, pd.Series)
            or isinstance(other, pd.Index)
        ):
            return NotImplemented
        # rely on Quantity comparison that will return a boolean array
        return self._data == other._data

    @property
    def nbytes(self) -> int:
        """The byte size of the data."""
        return self._itemsize * len(self)

    def isna(self):
        """A 1-D array indicating if each value is missing."""
        return unumpy.isnan(self._data)  # type: ignore

    def take(self, indexer, allow_fill: bool = False, fill_value: Any = None):
        """Take elements from an array.

        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        # return self._from_sequence(result, dtype=UfloatDtype)
        return UfloatArray(result)

    def copy(self):
        """
        Return a copy of the array.
        Returns
        -------
        UfloatArray
        """
        return type(self)(self._data.copy(), self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Iterable[UfloatArrayT]):
        """Concatenate multiple arrays."""
        return cls(np.concatenate([x._data for x in to_concat]))

    def dropna(self):
        """
        Return UfloatArray without NA values.
        Returns
        -------
        valid : UfloatArray
        """
        return type(self)(self[~self.isna()], self._dtype)

    def unique(self):
        """
        Compute the UfloatArray of unique values.
        Returns
        -------
        uniques : UfloatArray
        """
        from pandas.core.algorithms import unique

        uniques = unique(self._data)
        # return self._from_sequence(uniques, dtype=UfloatDtype)
        return UfloatArray(uniques)

    def searchsorted(self, value, side="left", sorter=None):
        return np.searchsorted(self._data, value, side=side, sorter=sorter)  # type: ignore

    def _values_for_argsort(self):
        return np.array(self._data)

    def _reduce(self, name, *, skipna=True, **kwargs):
        functions = {
            "all": all,
            "any": any,
            "min": min,
            "max": max,
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "prod": np.prod,
            # TODO implement for uncertainties
            # "std": lambda x:np.std(x, ddof=1),
            # "var": lambda x:np.var(x, ddof=1),
            # "sem": lambda x:np.std(x, ddof=0),
            # "kurt": lambda x:scipy.stats.kurtosis(x, bias=False),
            # "skew": lambda x:scipy.stats.skew(x, bias=False),
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            quantity = self.dropna()._data
        else:
            quantity = self._data

        return functions[name](quantity)

    def _formatter(self, boxed: bool = False):
        def formatting_function(uflt):
            return "{}".format(uflt)

        return formatting_function

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool = True):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype :  bool
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype
            (default True)
        Returns
        -------
        A method that can be bound to a method of a class
        Example
        -------
        Given an ExtensionArray subclass called MyExtensionArray, use
        >>> __add__ = cls._create_method(operator.add)
        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """
        from pandas.compat import set_function_name

        def _binop(self, other):
            def validate_length(obj1, obj2):
                # validates length and converts to listlike
                try:
                    if len(obj1) == len(obj2):
                        return obj2
                    else:
                        raise ValueError("Lengths must match")
                except TypeError:
                    return [obj2] * len(obj1)

            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param, cls):
                    return param._data
                elif isinstance(param, AffineScalarFunc):
                    return param
                elif is_list_like(param) and isinstance(param[0], AffineScalarFunc):
                    return param
                else:
                    return param

            if isinstance(other, (pd.Series, pd.DataFrame)):
                return NotImplemented
            lvalues = self._data
            other = validate_length(lvalues, other)
            rvalues = convert_values(other)
            # Pint quantities may only be exponented by single values, not arrays.
            # Reduce single value arrays to single value to allow power ops
            # if isinstance(rvalues, AffineScalarFunc):
            #     if len(set(np.array(rvalues.data))) == 1:
            #         rvalues = rvalues[0]
            # elif len(set(np.array(rvalues))) == 1:
            #     rvalues = rvalues[0]
            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            # print(f"{lvalues =}")
            # print(f"{rvalues =}")
            res = op(lvalues, rvalues)

            # if op.__name__ == "divmod":
            #     return (
            #         cls.from_1darray_quantity(res[0]),
            #         cls.from_1darray_quantity(res[1]),
            #     )

            if coerce_to_dtype:
                try:
                    # res = cls.from_1darray_quantity(res)
                    # enforce a Quantity object when a calculation leads to dimensionless
                    # hence physipy returns an array, not a dimensionless quantity (df["a"]/df["a"])
                    # res = UArray(res)
                    res = cls(res)
                except TypeError:
                    pass

            return res

        op_name = f"__{op}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        # return cls._create_method(op)
        return cls._create_method(op, coerce_to_dtype=True)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)

    # @classmethod
    # def from_1darray_quantity(cls, quantity):
    #     if not is_list_like(quantity.value):
    #         raise TypeError("quantity's magnitude is not list like")
    #     return cls(quantity)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Series implements __array_ufunc__. As part of the implementation, pandas unboxes the ExtensionArray
        from the Series, applies the ufunc, and re-boxes it if necessary.

        If applicable, we highly recommend that you implement __array_ufunc__ in your extension array to avoid
        coercion to an ndarray. See the NumPy documentation for an example.

        As part of your implementation, we require that you defer to pandas when a pandas container (Series, DataFrame, Index)
        is detected in inputs. If any of those is present, you should return NotImplemented. pandas will take care of unboxing
        the array from the container and re-calling the ufunc with the unwrapped input.
        """
        if any(
            map(
                lambda x: isinstance(x, pd.DataFrame) or isinstance(x, pd.Series),
                inputs,
            )
        ):
            raise NotImplementedError
        if not (method == "__call__" or method == "reduce"):
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )

        # first compute the raw quantity result
        inputs = map(lambda x: x.quantity, inputs)
        if method == "__call__":
            qres = ufunc.__call__(*inputs)
        elif method == "reduce":
            qres = ufunc.reduce(*inputs, **kwargs)
        else:
            raise NotImplementedError

        # then wrap it in a QuantityArray, with the corresponding dtype
        # good thing is that the dimensions work is handled in Quantity
        try:
            n = len(qres)
            return UfloatArray(qres)
        except Exception as e:
            return qres

    def __array__(self, dtype=None, copy=False):
        # if dtype is None or is_object_dtype(dtype):
        #    return self._to_array_of_quantity(copy=copy)
        # if (isinstance(dtype, str) and dtype == "string") or isinstance(
        #    dtype, pd.StringDtype
        # ):
        #    return pd.array([str(x) for x in self.quantity], dtype=pd.StringDtype())
        # if is_string_dtype(dtype):
        #    return np.array([str(x) for x in self.quantity], dtype=str)
        return np.array(self._data, dtype=dtype, copy=copy)


UfloatArray._add_arithmetic_ops()
UfloatArray._add_comparison_ops()


@register_series_accessor("u")
class UfloatSeriesAccessor:
    def __init__(self, series):
        # self._validate(series)
        self._obj = series
        self._asuarray = UArray(series)

    # @staticmethod
    # def _validate(obj):
    #     if obj.dtype != "ufloat":
    #         raise TypeError("Dtype has to be 'ufloat' in order to use this accessor!")

    @property
    def n(self):
        return pd.Series(self._asuarray.n, index=self._obj.index, name=self._obj.name)

    @property
    def s(self):
        return pd.Series(self._asuarray.s, index=self._obj.index, name=self._obj.name)


@register_dataframe_accessor("u")
class UfloatDataFrameAccessor:
    def __init__(self, dataframe):
        # self._validate(dataframe)
        self._obj = dataframe
        self._asuarray = UArray(dataframe)

    # @staticmethod
    # def _validate(obj):
    #     #! TODO
    #     try:
    #         UArray(obj)
    #     except Exception as e:
    #         raise e

    @property
    def n(self):
        return pd.DataFrame(
            self._asuarray.n, index=self._obj.index, columns=self._obj.columns
        )

    @property
    def s(self):
        return pd.DataFrame(
            self._asuarray.s, index=self._obj.index, columns=self._obj.columns
        )

    @property
    def sep(self):
        df = pd.DataFrame(data=None, index=self._obj.index)
        for column_name in self._obj:
            if isinstance(self._obj[column_name].iloc[0], AffineScalarFunc):
                series_n = self._obj[column_name].astype("ufloat").u.n
                series_s = self._obj[column_name].astype("ufloat").u.s
                # TODO: change 'd' to PREFIX
                series_s.name = f"d{series_s.name}"
                df = pd.concat([df, series_n, series_s], axis=1)
            else:
                df = pd.concat([df, self._obj[column_name]], axis=1)

        return df

    @property
    def com(self):

        df = pd.DataFrame(data=None, index=self._obj.index)

        errors = []
        errored = []
        for column_name in self._obj:
            shortened = column_name[1:]
            # TODO: PREFIX
            if column_name.startswith("d") and shortened in self._obj.columns:
                errors.append(column_name)
                errored.append(shortened)

        for column_name in self._obj:
            if column_name in errored:
                continue
            elif column_name in errors:
                shortened = column_name[1:]
                if is_complex_dtype(self._obj[shortened]):
                    # print(self._obj[shortened])
                    # print(abs(self._obj[shortened]))
                    self._obj[shortened] = abs(self._obj[shortened])

                df = pd.concat(
                    [
                        df,
                        pd.Series(
                            uarray(self._obj[shortened], self._obj[column_name]),
                            name=shortened,
                        ).astype("ufloat"),
                    ],
                    axis=1,
                )
            else:
                df = pd.concat([df, self._obj[column_name]], axis=1)

        return df
