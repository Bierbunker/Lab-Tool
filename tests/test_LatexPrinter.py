from labtool_ex2 import Project
from labtool_ex2 import Symbol
from sympy import symbols


def test_basic_symbol_printing():
    gm = {
        "t": r"t",
        "tau": r"\tau",
    }
    gv = {
        "t": r"\si{\second}",
        "tau": r"\si{\second}",
    }
    P = Project(None, global_variables=gv, global_mapping=gm, font=13)
    P.print_expr(t)
    x = Symbol("x")
    P.print_expr(x)
