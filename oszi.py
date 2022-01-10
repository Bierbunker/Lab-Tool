import sympy as simp
from sympy import sin, pi
import numpy as np
import pandas as pd
from labtool_ex2 import Project
from uncertainties import ufloat
from uncertainties import umath


if __name__ == "__main__":
    # Preamble and configuration
    gm = {
        "t": r"t",
        "tau": r"\tau",
        "I": r"I",
        "I0": r"I_0",
        "lI0": r"\ln{(I_0)}",
        "UC": r"U_C",
        "lI": r"\ln{(I)}",
        "lUC": r"\ln{(U_C)}",
        "U0": r"U_0",
        "lU0": r"\ln{(2U_0)}",
        "UR": r"U_R",
        "R": r"R_L",
    }
    gv = {
        "t": r"\si{\second}",
        "tau": r"\si{\second}",
        "I": r"\si{\ampere}",
        "I0": r"\si{\ampere}",
        "lI0": r"$\ln{(\si{\ampere})}$",
        "UC": r"\si{\volt}",
        "lI": r"$\ln{(\si{\ampere})}$",
        "lUC": r"$\ln{(\si{\volt})}$",
        "U0": r"\si{\volt}",
        "lU0": r"$\ln{(\si{\volt})}$",
        "UR": r"\si{\volt}",
        "R": r"\si{\ohm}",
    }
    simp.var(list(gv))
    P = Project("Oszi", global_variables=gv, global_mapping=gm, font=13)
    ax = P.figure.add_subplot()

    # Versuch 1
    P.load_data("./Data/Oszi/versuch.csv")
    print(P.data)

    # Ladekurve Spannung 1
    P.data = P.data[P.data["t"].between(7.5999773e-03, 1.6840045e-02, inclusive=False)]
    P.data["t"] = P.data["t"] - P.data["t"].values[0]
    IR = UR / R
    # P.data["I"] = P.data["UR"] / P.data["R"]
    DUC = 0.03 * UC + 0.1 * 0.074 + 0.001
    DUR = 0.03 * UR + 0.1 * 0.142 + 0.001
    P.data["dUC"] = P.apply_df(DUC)
    P.data["dUR"] = P.apply_df(DUR)
    P.data["I"] = P.apply_df(IR)
    P.data["dI"] = P.apply_df_err(IR)
    P.data["lI"] = np.log(-P.data["I"] + 1.001 * P.data["I"].max())
    P.data["lUC"] = np.log(P.data["UC"] - 1.001 * P.data["UC"].min())
    P.data["dlI"] = (
        np.abs(1 / (-P.data["I"] + 1.001 * P.data["I"].max())) * P.data["dI"]
    )
    P.data["dlUC"] = (
        np.abs(1 / (P.data["UC"] - 1.001 * P.data["UC"].min())) * P.data["dUC"]
    )
    print(P.data[["lI", "lUC"]])
    # lI = -t/tau + simp.log(I0)
    lI = -t / tau + lI0
    lUC = -t / tau + lU0

    UC = 2 * U0 * simp.exp(-t / tau) - U0
    I = -I0 * simp.exp(-t / tau)
    P.plot_data(
        ax,
        "t",
        "UC",
        label="Spannungsdaten",
        style="b",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="UC",
        eqn=UC,
        style=r"#1cb2f5",
        label="Spannungskurven",
        offset=[30, 10],
        use_all_known=False,
        guess={"tau": 0.001, "U0": 0.25},
        bounds=[
            {"name": "U0", "min": "0", "max": "5"},
            {"name": "tau", "min": "0", "max": "0.05"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    P.data["t"] = P.data["t"] - P.data["t"].values[0]
    P.plot_data(
        ax,
        "t",
        "I",
        label="Stromdaten",
        style="y",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="I",
        eqn=I,
        style=r"#FF8300",
        label="Stromkurven",
        use_all_known=False,
        offset=[30, 70],
        guess={"tau": 0.001, "I0": 0.0004},
        bounds=[
            {"name": "I0", "min": "0.0001", "max": "0.0005"},
            {"name": "tau", "min": "0.0005", "max": "0.005"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    ax.set_title(
        r"Spannungs- und Stromentladungskurve am Kondensator @\SI{1}{\micro\farad}"
    )
    P.ax_legend_all(loc=7)
    P.figure.set_tight_layout(True)
    ax = P.savefig(f"entladekurve.png", clear=True)
    # Linearisierung Entladekurve
    P.data = P.data[P.data["t"].between(0, 0.004, inclusive=False)]
    P.plot_data(
        ax,
        "t",
        "lUC",
        label="Spannungsdaten",
        style="b",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="lUC",
        eqn=lUC,
        style=r"#1cb2f5",
        label="Spannungskurven",
        offset=[0, 0],
        use_all_known=False,
        guess={"tau": 0.001, "lU0": 0.25},
        bounds=[
            {"name": "lU0", "min": "0.1", "max": "5"},
            {"name": "tau", "min": "0.0001", "max": "0.05"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    P.plot_data(
        ax,
        "t",
        "lI",
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="lI",
        eqn=lI,
        style=r"#FF8300",
        label="Stromkurven",
        use_all_known=False,
        offset=[0, 25],
        guess={"tau": 0.001, "lI0": 0.0025},
        bounds=[
            {"name": "lI0", "min": "0.001", "max": "0.5"},
            {"name": "tau", "min": "0.0009", "max": "0.005"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    ax.set_title(
        r"Spannungs- und Stromentladungskurve \\ am Kondensator @\SI{1}{\micro\farad} Linearisierung"
    )
    P.ax_legend_all(loc=1)
    P.figure.set_tight_layout(True)
    ax = P.savefig(f"entladekurve_linear.png", clear=True)

    # Ladekurve Spannung 1
    simp.var(list(gv))
    P.data = P.dfs["versuch"][
        P.dfs["versuch"]["t"].between(-2.2800101e-03, 7.5999772e-03, inclusive=False)
    ]
    P.data["t"] = P.data["t"] - P.data["t"].values[0]
    IR = UR / R
    # P.data["I"] = P.data["UR"] / P.data["R"]
    DUC = 0.03 * UC + 0.1 * 0.074 + 0.001
    DUR = 0.03 * UR + 0.1 * 0.142 + 0.001
    P.data["dUC"] = P.apply_df(DUC)
    P.data["dUR"] = P.apply_df(DUR)
    P.data["I"] = P.apply_df(IR)
    P.data["dI"] = P.apply_df_err(IR)
    P.data["lI"] = np.log(P.data["I"])
    P.data["lUC"] = np.log(-P.data["UC"] + P.data["UC"].max())
    P.data["dlI"] = np.abs(1 / (P.data["I"])) * P.data["dI"]
    P.data["dlUC"] = np.abs(1 / (-P.data["UC"] + P.data["UC"].max())) * P.data["dUC"]

    print(P.data)
    P.plot_data(
        ax,
        "t",
        "UC",
        label="Spannungsdaten",
        style="b",
    )

    UC = -(2 * U0 * simp.exp(-t / tau) - U0)
    I = I0 * simp.exp(-t / tau)
    P.plot_fit(
        axes=ax,
        x="t",
        y="UC",
        eqn=UC,
        style=r"#1cb2f5",
        label="Spannungkurven",
        use_all_known=False,
        offset=[30, 70],
        guess={"tau": 0.001, "U0": 0.25},
        bounds=[
            {"name": "U0", "min": "0", "max": "5"},
            {"name": "tau", "min": "0", "max": "0.05"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    print(P.data)
    P.data = P.data.iloc[1:, :]
    print(P.data)
    P.plot_data(
        ax,
        "t",
        "I",
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="I",
        eqn=I,
        style=r"#FF8300",
        label="Stromkurven",
        use_all_known=False,
        offset=[30, 10],
        guess={"tau": 0.001, "I0": 0.0004},
        bounds=[
            {"name": "I0", "min": "0.0001", "max": "0.0006"},
            {"name": "tau", "min": "0.0005", "max": "0.002"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    ax.set_title(
        r"Spannungs- und Stromladungskurve am Kondensator @\SI{1}{\micro\farad}"
    )
    P.ax_legend_all(loc=7)
    P.figure.set_tight_layout(True)
    ax = P.savefig(f"ladekurve.png")

    # Linearisierung Ladekurve
    P.data = P.data[P.data["t"].between(0, 0.004, inclusive=False)]
    # lI = -t/tau + simp.log(I0)
    lI = -t / tau + lI0
    lUC = -t / tau + lU0
    P.plot_data(
        ax,
        "t",
        "lUC",
        label="Spannungsdaten",
        style="b",
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="lUC",
        eqn=lUC,
        style=r"#1cb2f5",
        label="Spannungskurven",
        offset=[0, 0],
        use_all_known=False,
        guess={"tau": 0.001, "lU0": 0.25},
        bounds=[
            {"name": "lU0", "min": "0.1", "max": "5"},
            {"name": "tau", "min": "0.0001", "max": "0.05"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    P.plot_data(
        ax,
        "t",
        "lI",
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,
        x="t",
        y="lI",
        eqn=lI,
        style=r"#FF8300",
        label="Stromkurven",
        use_all_known=False,
        offset=[0, 25],
        guess={"tau": 0.001, "lI0": 0.0025},
        bounds=[
            {"name": "lI0", "min": "0.001", "max": "0.5"},
            {"name": "tau", "min": "0.0001", "max": "0.005"},
        ],
        add_fit_params=True,
        granularity=10000,
        gof=True,
    )
    ax.set_title(
        r"Spannungs- und Stromladungskurve \\ am Kondensator @\SI{1}{\micro\farad} Linearisierung"
    )
    P.ax_legend_all(loc=1)
    P.figure.set_tight_layout(True)
    # Beim wegschneiden der daten auf auflösungsvermögen schieben
    ax = P.savefig(f"ladekurve_linear.png", clear=True)
#    # Entladekurve Strom 1
#    simp.var(list(gv))
#    P.data = P.dfs["versuch"][P.dfs["versuch"]['t'].between(-2.2800101e-03, 7.5999772e-03, inclusive=False)]
#    P.data["t"] = P.data["t"] - P.data["t"].values[0]
#    print(P.data)
#    P.plot_data(
#        ax,
#        "t",
#        "UC",
#        label="Gemessene Daten",
#        style="y",
#    )
#
#    UC = -(2*U0*simp.exp(-t/tau) - U0)
#    P.plot_fit(
#        axes=ax,
#        x="t",
#        y="UC",
#        eqn=UC,
#        style="r-",
#        label="Ladungskurve",
#        use_all_known=False,
#        offset=[30, 10],
#        guess={"tau": 0.001, "U0": 0.25},
#        bounds=[
#            {"name": "U0", "min": "0", "max": "5"},
#            {"name": "tau", "min": "0", "max": "0.05"},
#        ],
#        add_fit_params=True,
#        granularity=10000,
#    )
#    ax.set_title(r"Spannungs Ladungskurve am Kondensator @\SI{1}{\micro\farad}")
#    ax = P.savefig(f"ladekurve.png")
