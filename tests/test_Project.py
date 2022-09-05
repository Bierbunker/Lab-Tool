from labtool_ex2 import Project
from labtool_ex2 import Symbol
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
    P = Project(None, global_variables=gv, global_mapping=gm, font=13)
    assert str(t) == "t"
    assert str(tau) == "tau"
    assert type(t) == Symbol
    assert type(tau) == Symbol
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


def test_symbol():
    gm = {
        "t": r"t",
        "tau": r"\tau",
    }
    gv = {
        "t": r"\si{\second}",
        "tau": r"\si{\second}",
    }
    p = Project(None, global_variables=gv, global_mapping=gm, font=13)
    filepath = os.path.join(os.path.dirname(__file__), "./data/input/short_test.csv")
    p.load_data(filepath)

    summe = t + tau
    assert str(summe) == "t + tau"
    p.resolve(summe)
    from inspect import currentframe

    frame = currentframe()  # type: ignore
    assert summe.name in frame.f_locals and summe.name in frame.f_globals  # type:ignore
    assert type(summe) is Symbol
    assert summe.name == "summe"
    assert all(summe.data == p.data.t + p.data.tau)
    assert hasattr(p.data, "summe")


def test_usymbol():
    gm = {
        "t": r"t",
        "tau": r"\tau",
    }
    gv = {
        "t": r"\si{\second}",
        "tau": r"\si{\second}",
    }
    P = Project(None, global_variables=gv, global_mapping=gm, font=13)
    filepath = os.path.join(os.path.dirname(__file__), "./data/input/short_test.csv")
    P.load_data(filepath)
    P.data = P.data.u.com

    summe = t + tau
    assert str(summe) == "t + tau"
    P.resolve(summe)
    from inspect import currentframe

    frame = currentframe()  # type: ignore
    assert summe.name in frame.f_locals and summe.name in frame.f_globals  # type:ignore
    assert isinstance(summe, Symbol)
    assert summe.name == "summe"
    assert all(summe.data == P.data.t + P.data.tau)
    assert hasattr(P.data, "summe")


# test for adding ufloat array stuff
# test_creation()
# test_symbol()
