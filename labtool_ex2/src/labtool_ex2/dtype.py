from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.arrays.base import ExtensionArray
from uncertainties.core import AffineScalarFunc, Variable
import numpy as np

from typing import Optional, List, Tuple, Union


class UfloatDtype(Variable, ExtensionDtype):
    _is_bool = True
    type = Variable

    @property
    def _is_numeric(self) -> bool:
        return True

    @property
    def na_value(self) -> Variable:
        """
        Return the genotype with variant information but no alleles specified
        """
        return Variable(value=0, std_dev=0)

    @property
    def name(self) -> str:
        return str(self)

    def __init__(self, value, std_dev, tag=None):
        super(UfloatDtype, self).__init__(value=value, std_dev=std_dev, tag=tag)

    # ExtensionDtype Methods
    # -------------------------
    @classmethod
    def construct_array_type(cls) -> type:
        """
        Return the array type associated with this dtype
        Returns
        -------
        type
        """
        return UfloatArray


class UfloatArray(ExtensionArray):
    def __init__(
        self,
        values: Union[list[Variable], "UfloatArray", np.ndarray],
        dtype: Optional[UfloatDtype] = None,
        copy: bool = False,
    ):
        values = _to_ip_array(values)  # TODO: avoid potential copy
        # TODO: dtype?
        if copy:
            values = values.copy()
        self.data = values

        # if std_devs is None:  # Obsolete, single tuple argument call
        #     (nominal_values, std_devs) = nominal_values

        self.data = np.vectorize(
            # ! Looking up uncert_core.Variable beforehand through
            # '_Variable = uncert_core.Variable' does not result in a
            # significant speed up:
            lambda v, s: Variable(v, s),
            otypes=[UfloatDtype] * len(values),
        )(nominal_values, std_devs)
