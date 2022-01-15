# 3rd party
import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    register_extension_dtype,
    register_series_accessor,
    register_dataframe_accessor,
)
from uncertainties.core import AffineScalarFunc, Variable
from uncertainties.unumpy import uarray

# own project imports
from .uarray import UArray



@register_extension_dtype
class UfloatDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray"""
    
    type = Variable  # type: ignore
    # linter wants a property, but pandas validates the type of name
    # as a string via assertion, therefore: AssertionError if written
    # as a property
    name = "ufloat"  # type: ignore
    _is_numeric = True
    _is_boolean = True
       
    
    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype
        Return
        -------
        type
        """
        return UfloatArray
    
    
class UfloatArray(ExtensionArray):
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
    
    def __init__(self, variables, dtype=None, copy=False):
        self._data = UArray(variables)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        return cls(scalars, dtype=dtype, copy=copy)
    
    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values)
    
    def __getitem__(self, item):
        """Select a subset of self."""
        return self._data[item]
    
    def __len__(self) -> int:
        """Length of this array."""
        return len(self._data)
    
    @property
    def _itemsize(self):
        from sys import getsizeof as sizeof
        return sizeof(Variable)
    
    @property
    def nbytes(self):
        """The byte size of the data."""
        return self._itemsize * len(self)
    
    @property
    def dtype(self):
        """An instance of 'ExtensionDtype'."""
        return UfloatDtype()
    
    def isna(self):
        """A 1-D array indicating if each value is missing."""
        return np.array([isinstance(x, Variable) for x in self._data], dtype=bool)
    
    def take(self, indexer, allow_fill=False, fill_value=None):
        """Take elements from an array.

        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)
    
    def copy(self):
        """Return a copy of the array."""
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(np.concatenate([x._data for x in to_concat]))


@register_series_accessor("u")
class UfloatSeriesAccessor:
    
    def __init__(self, series):
        #self._validate(series)
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
        #self._validate(dataframe)
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
        return pd.DataFrame(self._asuarray.n, index=self._obj.index, columns=self._obj.columns)
    
    @property
    def s(self):
        return pd.DataFrame(self._asuarray.s, index=self._obj.index, columns=self._obj.columns)
    
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
                df = pd.concat([df, pd.Series(uarray(self._obj[shortened], self._obj[column_name]), name=shortened)], axis=1)
            else:
                df = pd.concat([df, self._obj[column_name]], axis=1)
        
        return df
            
        #if not re.match(r"^d(\w)+(\.\d+)?$", column_name):
