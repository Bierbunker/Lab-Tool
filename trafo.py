import sympy as simp
from sympy import sin, pi
import pandas as pd
from labtool_ex2 import Project
from uncertainties import ufloat


if __name__ == "__main__":
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
    simp.var(list(gv))
    P = Project("Trafo", global_variables=gv, global_mapping=gm, font=13)
    ax = P.figure.add_subplot()

    # Versuch 1
    P.load_data("./Data/Trafo/versuch1.csv")
    print(P.data)
    P.print_table(
        P.data[["P1", "dP1", "U", "dU", "U1", "dU1", "I1", "dI1", "U2", "dU2"]],
        name="messwertversuch1",
    )
    S = U1 * I1
    P.data["S"] = P.apply_df(S)
    P.data["dS"] = P.apply_df_err(S)
    Q1 = (S ** 2 - P1 ** 2) ** 0.5
    P.data["Q1"] = P.apply_df(Q1)
    P.data["dQ1"] = P.apply_df_err(Q1)
    l = P1 / S
    P.data["l"] = P.apply_df(l)
    P.data["dl"] = P.apply_df_err(l)
    P.print_table(
        P.data[["S", "dS", "Q1", "dQ1", "l", "dl"]],
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
    simp.var(list(gv))
    P.load_data("./Data/Trafo/versuch2.csv", loadnew=True)
    print(P.data)
    P.print_table(
        P.data[
            [
                "P1",
                "dP1",
                "U1",
                "dU1",
                "I1",
                "dI1",
                "U2",
                "dU2",
                "I2",
                "dI2",
            ]
        ],
        name="messwertversuch2",
    )
    S = U1 * I1
    P.data["S"] = P.apply_df(S)
    P.data["dS"] = P.apply_df_err(S)
    Q1 = (S ** 2 - P1 ** 2) ** 0.5
    P.data["Q1"] = P.apply_df(Q1)
    P.data["dQ1"] = P.apply_df_err(Q1)
    l = P1 / S
    P.data["l"] = P.apply_df(l)
    P.data["dl"] = P.apply_df_err(l)
    P2 = U2 * I2
    P.data["P2"] = P.apply_df(P2)
    P.data["dP2"] = P.apply_df_err(P2)
    PV = P1 - P2
    P.data["PV"] = P.apply_df(PV)
    P.data["dPV"] = P.apply_df_err(PV)
    n = P2 / P1 * 100
    P.data["n"] = P.apply_df(n)
    P.data["dn"] = P.apply_df_err(n)
    P.print_table(
        P.data[
            ["S", "dS", "Q1", "dQ1", "l", "dl", "P2", "dP2", "PV", "dPV", "n", "dn"]
        ],
        name="wertversuch2",
    )

    # Versuch 3
    P.load_data("./Data/Trafo/versuch3.csv", loadnew=True)
    R = Ur / I2
    PR = Ur * I2
    P.data["R"] = P.apply_df(R)
    P.data["dR"] = P.apply_df_err(R)
    P.data["PR"] = P.apply_df(PR)
    P.data["dPR"] = P.apply_df_err(PR)
    P.print_table(
        P.data[
            [
                "P1",
                "dP1",
                "U",
                "dU",
                "U1",
                "dU1",
                "I1",
                "dI1",
            ]
        ],
        name="messwertversuch3_1",
    )
    P.print_table(
        P.data[
            [
                "U2",
                "dU2",
                "I2",
                "dI2",
                "Ur",
                "dUr",
            ]
        ],
        name="messwertversuch3_2",
    )
    P.print_table(
        P.data[["PR", "dPR", "R", "dR"]],
        name="wertversuch3",
    )
    S = U1 * I1
    P.data["S"] = P.apply_df(S)
    P.data["dS"] = P.apply_df_err(S)
    Q1 = (S ** 2 - P1 ** 2) ** 0.5
    P.data["Q1"] = P.apply_df(Q1)
    P.data["dQ1"] = P.apply_df_err(Q1)
    l = P1 / S
    P.data["l"] = P.apply_df(l)
    P.data["dl"] = P.apply_df_err(l)
    P2 = U2 * I2
    P.data["P2"] = P.apply_df(P2)
    P.data["dP2"] = P.apply_df_err(P2)
    PV = P1 - PR
    P.data["PV"] = P.apply_df(PV)
    P.data["dPV"] = P.apply_df_err(PV)
    n = PR / P1 * 100
    P.data["n"] = P.apply_df(n)
    P.data["dn"] = P.apply_df_err(n)
    P.print_table(
        P.data[
            [
                "S",
                "dS",
                "Q1",
                "dQ1",
                "l",
                "dl",
                "P2",
                "dP2",
                "PV",
                "dPV",
                "n",
                "dn",
            ]
        ],
        name="wertversuch3_extra",
    )
    P.plot_data(
        ax,
        "R",
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
    simp.var(list(gv))
    print(P.data)
    PR = U2 ** 2 * R / (R ** 2 + (XL) ** 2)
    P.print_expr(PR)
    theoPR = U2 ** 2 * R / (R ** 2 + xl ** 2)
    P.data["theoPR"] = P.apply_df(theoPR)
    P.data["dtheoPR"] = P.apply_df_err(theoPR)
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
    df = pd.DataFrame(
        {"x": [ufloat(11, 1) * 1e8, ufloat(11, 1)], "y": [ufloat(11, 1), ufloat(11, 1)]}
    )
    print(df)
    # arr = unumpy.uarray([1, 2], [0.01, 0.002])
    P.print_ftable(df, name="test", split=True)

    print(test.__format__("").split(r"+/-"))

    ax.set_title(f"Leistungskurve am Lastwiderstand")
    P.ax_legend_all(loc=1)
    P.figure.savefig(f"./Output/{P.name}/leistungskurve.png", dpi=400)
    P.figure.clear()
