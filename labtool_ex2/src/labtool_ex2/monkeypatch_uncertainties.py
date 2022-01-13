"""Monkey patches for package 'uncertainties':
Uncertainties: a Python package for calculations with uncertainties,
Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["display", "undo"]

# std lib
from importlib import reload
from math import ceil

# 3rd party
import uncertainties.core as uc


def display() -> None:
    """Update uncertainties' formatting function to a convention used in
    'Einführung in die physikalischen Messmethoden' (EPM), scriptum version 7.
    """

    def EPM_precision(std_dev: float) -> tuple[int, float]:
        """Return the number of significant digits to be used for the given
        standard deviation, according to the rounding rules of EPM instead
        of PDG (Particle Data Group).
        Also return the effective standard deviation to be used for display
        """
        dig, _, s = _digits_exponent_std_dev(std_dev)
        return dig, s

    # ufloat is a factory function which instantiates a uncertainties.core.Variable
    # which inherits from uncertainties.core.AffineScalarFunc
    # uncertainties.core.PDG_precision is used for uncertainties.core.AffineScalarFunc.__format__
    # which is used for uncertainties.core.AffineScalarFunc.__str__
    # therefore changing the behavior of that function changes the way ufloats are diplayed
    uc.PDG_precision = EPM_precision

    # uncertainties.core.AffineScalarFunc.__repr__ does not round n and s,
    # instead displays them precisely
    # to represent all ufloats correctly rounded in containers
    uc.AffineScalarFunc.__repr__ = uc.AffineScalarFunc.__str__

    return None


def undo() -> None:
    """Reload uncertainties.core and remove applied monkey patches"""
    reload(uc)
    return None


def _digits_exponent_std_dev(std_dev: float) -> tuple[int, int, float]:
    """Find the amount of significant digits the exponent of those digits to the base 10.
    Also return the effective standard deviation.
    Provide data needed by function 'display' (subfunction 'EPM_precision') and
    function 'init' (subfunction 'round_conventional')
    Return one significant digit, except when this digit is 1, then return two.
    """

    if std_dev:  # std_dev != 0

        # exponent of base 10
        exponent = uc.first_digit(std_dev)

        # calculate mantissa of std_dev
        # round to 3 digits to minimize machine epsilon
        mantissa = round(std_dev * 10 ** (-exponent), 3)

        # significant digits to consider for rounding
        sig_digits = 1

        # two significant digits if first digit is 1, one digit otherwise
        if mantissa <= 1.9:
            sig_digits += 1
            exponent -= 1
            mantissa *= 10

        # round up according to significant digits
        s = ceil(mantissa) * 10 ** exponent

        return sig_digits, exponent, s

    else:  # std_dev == 0
        return 0, 0, 0.0
