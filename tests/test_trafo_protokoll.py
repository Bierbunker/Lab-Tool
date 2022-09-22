from sympy import sin, pi
from labtool_ex2 import Project
from uncertainties import ufloat
import os

# pyright: reportUndefinedVariable=false


def test_trafo_protokoll():
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
    P = Project("Trafo", global_variables=gv, global_mapping=gm, font=13)
    ax = P.figure.add_subplot()

    # Versuch 1
    filepath = os.path.join(
        os.path.dirname(__file__), "./data/input/trafo_versuch1.csv"
    )
    P.load_data(filepath)
    # print(P.data)
    P.print_table(
        "P1",
        "U",
        "U1",
        "I1",
        "U2",
        name="messwertversuch1",
    )
    S = U1 * I1
    P.resolve(S)
    Q1 = (S**2 - P1**2) ** 0.5
    P.resolve(Q1)
    l = P1 / S
    P.resolve(l)
    P.print_table(
        "S",
        "Q1",
        "l",
        name="wertversuch1",
    )
    # P2 = U2 * I2
    # P.data["P2"] = P.apply_df(P2)
    # P.data["dP2"] = P.apply_df_err(P2)
    # PV = P - P2
    # P.data["PV"] = P.apply_df(PV)
    # P.data["dPV"] = P.apply_df_err(PV)
    # n = P2 / P1
    # P.data["n"] = P.apply_df(n)
    # P.data["dn"] = P.apply_df_err(n)

    # Versuch 2
    P.vload()
    filepath = os.path.join(
        os.path.dirname(__file__), "./data/input/trafo_versuch2.csv"
    )
    P.load_data(filepath, loadnew=True)
    # print(P.data)
    P.print_table(
        "P1",
        "U1",
        "I1",
        "U2",
        "I2",
        name="messwertversuch2",
    )
    S = U1 * I1
    # print(P.data)
    P.resolve(S)
    Q1 = (S**2 - P1**2) ** 0.5
    P.resolve(Q1)
    P.data = P.data.u.com
    # print(P.data.dtypes)
    # print(P.data)
    P.data = P.data.u.sep
    # P.data["dQ1"] = P.apply_df_err(Q1)
    l = P1 / S
    P.resolve(l)
    P2 = U2 * I2
    P.resolve(P2)
    PV = P1 - P2
    P.resolve(PV)
    n = P2 / P1 * 100
    P.resolve(n)
    P.print_table(
        "S",
        "Q1",
        "l",
        "P2",
        "PV",
        "n",
        name="wertversuch2",
    )

    # Versuch 3
    filepath = os.path.join(
        os.path.dirname(__file__), "./data/input/trafo_versuch3.csv"
    )
    P.load_data(filepath, loadnew=True)
    R = Ur / I2
    PR = Ur * I2
    P.resolve(R)
    P.resolve(PR)
    P.print_table(
        "P1",
        "U",
        "U1",
        "I1",
        name="messwertversuch3_1",
    )
    P.print_table(
        "U2",
        "I2",
        "Ur",
        name="messwertversuch3_2",
    )
    P.print_table(
        "PR",
        "R",
        name="wertversuch3",
    )
    S = U1 * I1
    P.resolve(S)
    Q1 = (S**2 - P1**2) ** 0.5
    P.resolve(Q1)
    l = P1 / S
    P.resolve(l)
    P2 = U2 * I2
    P.resolve(P2)
    PV = P1 - PR
    P.resolve(PV)
    n = PR / P1 * 100
    P.resolve(n)
    P.print_table(
        "S",
        "Q1",
        "l",
        "P2",
        "PV",
        "n",
        name="wertversuch3_extra",
        split=True,
    )
    P.plot_data(
        ax,
        R,
        "PR",
        label="Gemessene Daten",
        style="r",
        errors=True,
    )
    # P.plot_data(
    #     ax,
    #     "R",
    #     "P2",
    #     label="Nice",
    #     style="r",
    #     errors=True,
    # )
    P.vload()
    PR = U2**2 * R / (R**2 + (XL) ** 2)
    P.print_expr(PR)
    theoPR = U2**2 * R / (R**2 + xl**2)
    P.resolve(theoPR)
    P.plot_function(
        axes=ax,
        x="R",
        y="theoPR",
        expr=theoPR,
        label="theo. Leistungskurven",
        style="green",
        errors=True,
    )

    P.plot_fit(
        axes=ax,
        x="R",
        y="PR",
        eqn=PR,
        style="r",
        label="Leistungskurven",
        use_all_known=False,
        offset=[30, 10],
        guess={"U2": 24, "XL": 31},
        bounds=[
            {"name": "U2", "min": "0", "max": "25"},
            {"name": "L", "min": "30", "max": "32"},
        ],
        add_fit_params=True,
        granularity=10000,
    )
    test = ufloat(68, 1.4)
    # df = pd.DataFrame(
    #     {"x": [ufloat(11, 1) * 1e8, ufloat(11, 1)], "y": [ufloat(11, 1), ufloat(11, 1)]}
    # )
    # print(df)
    # arr = unumpy.uarray([1, 2], [0.01, 0.002])
    # P.print_ftable(df, name="test", split=True)

    # print(test.__format__("").split(r"+/-"))

    ax.set_title(f"Leistungskurve am Lastwiderstand")
    P.ax_legend_all(loc=1)
    ax = P.savefig(f"leistungskurve.pdf")
