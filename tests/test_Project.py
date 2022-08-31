from labtool_ex2 import Project
from labtool_ex2 import Symbol
from pandas import DataFrame
from sympy import symbols
import os

import pytest

# pyright: reportUndefinedVariable=false
# flake8: ignore F821


def test_c_symbol():
    x = Symbol("x", project=Project(None))

    # x = BSymbol(["x",], )
    symbols(["f", "g", "h"], cls=Symbol, project=Project(None))
    # P = Project("Test", global_variables=gv, global_mapping=gm, font=13)


def test_creation():
    gm = {
        "t": r"t",
        "tau": r"\tau",
    }
    gv = {
        "t": r"\si{\second}",
        "tau": r"\si{\second}",
    }
    P = Project("Test", global_variables=gv, global_mapping=gm, font=13)
    assert str(t) == "t"
    assert str(tau) == "tau"
    assert type(t) == Symbol
    assert type(tau) == Symbol
    print(dir(t))
    # print(type(t))
    with pytest.raises(KeyError) as exc_info:
        t.data
    with pytest.raises(KeyError) as exc_info:
        assert not hasattr(t, "data")
    filepath = os.path.join(os.path.dirname(__file__), "./data/input/short_test.csv")
    P.load_data(filepath)
    # print(f"{P.data['t'] =}")
    print(f"{P.data =}")
    print(t.data)
    assert hasattr(t, "data")
    assert all(t.data == P.data["t"])


# test_creation()
# test_c_symbol()
