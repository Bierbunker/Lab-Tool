# standard library
from __future__ import annotations

# 3rd party
import numpy as np
import pandas as pd
from uncertainties.core import Variable
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.arrays.base import ExtensionArray
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor


@register_extension_dtype
class UfloatDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray"""
     
    @property
    def type(self):
        return Variable
    
    @property
    def name(self):
        return "ufloat"

    @property
    def _is_numeric(self):
        return True
    
    @property
    def _is_boolean(self):
        return True
    
    #_metadata = ("nominal_value", "std_dev", "tag")
    
    
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
    

    def __init__(self, values, dtype=None, copy=False):
        uarray = []
        for value in values:
            try:
                uarray.append(Variable(value.nominal_value, value.std_dev, tag=value.tag))
            except AttributeError:
                uarray.append(Variable(value, 0))
                
        self._data = np.asarray(uarray, dtype=object)
        
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

        result = take(
            data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)
    
    def copy(self):
        """Return a copy of the array."""
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(np.concatenate([x._data for x in to_concat]))


@register_series_accessor("u")
class UfloatAccessor:
    def __init__(self, pandas_object):
        self._validate(pandas_object)
        self._obj = pandas_object
        self._n = pd.Series([x.n for x in pandas_object])
        self._s = pd.Series([x.s for x in pandas_object])
        
    @staticmethod
    def _validate(obj):
        try:
            obj.n; obj.s
        except AttributeError as e:
            raise e
        
    @property
    def n(self):
        return self._n
    
    @property
    def s(self):
        return self._s