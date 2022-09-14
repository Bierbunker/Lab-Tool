import sympy as simp
import os

from labtool_ex2 import Project

# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false

# if __name__ == "__main__":
def test_oszi_protokoll():
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
    P = Project("Oszi", global_variables=gv, global_mapping=gm, font=13)
    ax = P.figure.add_subplot()

    print(UC)
    print(UR)

    # Versuch 1
    filepath = os.path.join(os.path.dirname(__file__), "./data/input/oszi_versuch.csv")
    P.load_data(filepath)
    print(P.data)

    # Ladekurve Spannung 1
    P.data = P.data[P.data["t"].between(7.5999773e-03, 1.6840045e-02, inclusive=False)]
    P.data["t"] = P.data["t"] - P.data["t"].values[0]
    I = UR / R
    dUC = 0.03 * UC + 0.1 * 0.074 + 0.001
    dUR = 0.03 * UR + 0.1 * 0.142 + 0.001
    P.inject_err(dUC)
    P.inject_err(dUR)
    P.resolve(I)
    print(P.data)
    lI = simp.log(-I + 1.001 * I.data.max())
    lUC = simp.log(UC - 1.001 * UC.data.min())
    P.resolve(lI)
    P.resolve(lUC)
    print(P.data[["lI", "lUC"]])
    lI = -t / tau + lI0
    lUC = -t / tau + lU0

    UC = 2 * U0 * simp.exp(-t / tau) - U0
    I = -I0 * simp.exp(-t / tau)
    # change function signature
    P.plot_data(
        ax,
        t,
        UC,
        label="Spannungsdaten",
        style="b",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,
        x=t,
        y=UC,
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
        t,
        I,
        label="Stromdaten",
        style="y",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,
        x=t,
        y=I,
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
        ax,  # type: ignore
        t,
        lUC,
        label="Spannungsdaten",
        style="b",
        # errors=True,
    )
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=lUC,
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
        ax,  # type: ignore
        x=t,
        y=lI,
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=lI,
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
    ax.set_title(  # type: ignore
        r"Spannungs- und Stromentladungskurve \\ am Kondensator @\SI{1}{\micro\farad} Linearisierung"
    )
    P.ax_legend_all(loc=1)
    P.figure.set_tight_layout(True)
    ax = P.savefig(f"entladekurve_linear.png", clear=True)

    # Ladekurve Spannung 1
    # simp.var(list(gv))
    P.vload()
    P.data = P.dfs["oszi_versuch"][
        P.dfs["oszi_versuch"]["t"].between(
            -2.2800101e-03, 7.5999772e-03, inclusive=False
        )
    ]
    P.data["t"] = P.data["t"] - P.data["t"].values[0]
    I = UR / R
    dUC = 0.03 * UC + 0.1 * 0.074 + 0.001
    dUR = 0.03 * UR + 0.1 * 0.142 + 0.001
    P.inject_err(dUC)
    P.inject_err(dUR)
    P.resolve(I)
    lI = simp.log(I)
    lUC = simp.log(-UC + UC.data.max())
    P.print_expr(lUC)
    P.print_expr(lI)
    P.resolve(lI)
    P.resolve(lUC)

    print(P.data)
    P.plot_data(
        ax,  # type: ignore
        t,
        UC,
        label="Spannungsdaten",
        style="b",
    )

    UC = -(2 * U0 * simp.exp(-t / tau) - U0)
    I = I0 * simp.exp(-t / tau)
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=UC,
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
        ax,  # type: ignore
        t,
        I,
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=I,
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
    ax.set_title(  # type: ignore
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
        ax,  # type: ignore
        t,
        lUC,
        label="Spannungsdaten",
        style="b",
    )
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=lUC,
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
        ax,  # type: ignore
        t,
        lI,
        label="Stromdaten",
        style="y",
    )
    P.plot_fit(
        axes=ax,  # type: ignore
        x=t,
        y=lI,
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
    ax.set_title(  # type: ignore
        r"Spannungs- und Stromladungskurve \\ am Kondensator @\SI{1}{\micro\farad} Linearisierung"
    )
    P.ax_legend_all(loc=1)
    P.figure.set_tight_layout(True)
    # Beim wegschneiden der daten auf auflösungsvermögen schieben
    ax = P.savefig(f"ladekurve_linear.png", clear=True)


#    # Entladekurve Strom 1
#    simp.var(list(gv))
#    P.vload()
#    P.data = P.dfs["oszi_versuch"][P.dfs["oszi_versuch"]['t'].between(-2.2800101e-03, 7.5999772e-03, inclusive=False)]
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
