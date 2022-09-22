from labtool_ex2 import Project
import os

# pyright: reportUndefinedVariable=false


def test_simple_latex_export():
    gm = {
        "P1": r"P_1",
        "PR": r"P_R",
        "theoPR": r"P_R",
        "U1": r"U_1",
        "U2": r"U_2",
        "I1": r"I_1",
        "I2": r"I_2",
        "U": r"U",
        "Ur": r"U_r",
        "R": r"R_L",
        "S": r"S_1",
        "P2": r"P_2",
        "Q1": r"Q_1",
        "PV": r"P_V",
        "n": r"\eta",
        "l": r"\lambda",
        "a": r"a",
        "b": r"b",
        "c": r"c",
        "XL": r"X_L",
        "xl": r"x_l",
    }
    gv = {
        "P1": r"\si{\watt}",
        "PR": r"\si{\watt}",
        "theoPR": r"\si{\watt}",
        "U1": r"\si{\volt}",
        "U2": r"\si{\volt}",
        "I1": r"\si{\ampere}",
        "I2": r"\si{\ampere}",
        "U": r"\si{\volt}",
        "Ur": r"\si{\volt}",
        "R": r"\si{\ohm}",
        "S": r"\si{\va}",
        "P2": r"\si{\watt}",
        "Q1": r"\si{\var}",
        "PV": r"\si{\watt}",
        "n": r"\si{\percent}",
        "l": r"1",
        "a": r"1",
        "b": r"1",
        "c": r"1",
        "XL": r"\si{\ohm}",
        "xl": r"\si{\ohm}",
    }
    path = os.path.join(os.path.dirname(__file__), "data/input/trafo_versuch3.csv")
    P = Project("Test", global_variables=gv, global_mapping=gm, font=13)
    P.load_data(path)
    print(P.data)

    P.print_table(P1, U1, U, name="test")
    P.print_table(P1, U1, U, name="test", inline_units=True)
    P.print_table(P1, U1, U, name="test", split=True)
