from sympy import sin, pi
from labtool_ex2 import Project
from uncertainties import ufloat
import os

def test_error_func():
    gm = {
        "U": r"U",
        "I": r"I"
    }

    gv = {
        "U": r"U",
        "I": r"I"
    }

    P = Project("error_func", global_variables=gv, global_mapping=gm, font=13)
    #ax = P.figure.add_subplot()

    # Versuch 1


    filepath = os.path.join(
        os.path.dirname(__file__), "./data/input/error_func_test.csv"
    )
    P.load_data(filepath)

    P.print_table(
        "U",
        "I",
        name="messwertversuch1",
    )

    R = U / I

    P.resolve(R)

    S = R ** 2

    P.resolve(S)

    P.print_table(
        "R",
        "S",
        name="wertversuch1",
    )

