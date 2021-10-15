# import labtool_ex2 as lt

from labtool_ex2 import Equation
from labtool_ex2 import Project

if __name__ == "__main__":
    gm = {"na": "N.A.", "NA": "N.A.", "xi": r"\xi"}
    gv = {"NA": "1", "d": "mm"}
    P = Project("testing", global_variables=gv, global_mapping=gm, font=13)
    P.load_data("./src/labtool_ex2/Data/abbe/nadatablue")
    map = {"na": "N.A.", "xi": r"\xi"}
    na_vars = {"f": "mm", "r": "mm", "na": "1"}
    na = Equation(
        "r / f",
        na_vars,
        P.name,
        var_name="na",
        mapping=map,
        label="Numerische Apertur",
        dataframe=P.data[["r", "f", "dr", "df"]],
    )

    na.plot_fit(x="r", y="na", guess={"f": 1})
    # na.plot_fit(x="na", y="r", guess={"f": 1})
    d_vars = {"xi": "|||/mm", "d": "mm"}
    d = Equation(
        "1 / xi",
        d_vars,
        P.name,
        var_name="d",
        mapping=map,
        label="Auflösungsvermögen",
        dataframe=P.data[["xi", "dxi"]],
    )
    d.plot_data_scatter("xi", errors=True)
    P.data["d"] = d.apply_df()
    P.data["dd"] = d.apply_df_err()
    P.data["NA"] = na.apply_df()
    P.data["dNA"] = na.apply_df_err()
    P.equations["NA"] = na
    print(str(na._solve_for("f")))
    na.plot_data_scatter("r", errors=True)
    na.figure.clear()
    d_theoh_vars = {"NA": "1", "d": "mm"}
    d_theo_map = {"NA": "N.A."}
    d_theoh_blau = Equation(
        "0.61 * 10**-6 * 470/NA",
        d_theoh_vars,
        P.name,
        mapping=d_theo_map,
        figure=na.figure,
        var_name="d",
        label="Theo. Auflösungsvermögen Blau",
        dataframe=P.data[["NA"]],
    )
    d_theoh_blau.plot_function(x="NA", style="b-")
    d_theoh_red = Equation(
        "0.61 * 10**-6 * 635/NA",
        d_theoh_vars,
        P.name,
        mapping=d_theo_map,
        figure=na.figure,
        var_name="d",
        label="Theo. Auflösungsvermögen Rot",
        dataframe=P.data[["NA"]],
    )
    d_theoh_red.plot_function("NA")

    P.plot_data(
        x="NA",
        y="d",
        figure=na.figure,
        style="b",
        errors=True,
        labels=["N.A. / 1", "d / mm"],
    )
    P.load_data("./src/labtool_ex2/Data/abbe/nadatared", loadnew=True)
    d.data = P.data[["xi", "dxi"]]
    na.data = P.data[["r", "f", "dr", "df"]]
    P.data["d"] = d.apply_df()
    P.data["dd"] = d.apply_df_err()
    P.data["NA"] = na.apply_df()
    P.data["dNA"] = na.apply_df_err()
    P.plot_data(
        x="NA", y="d", figure=na.figure, errors=True, labels=["N.A. / 1", "d / mm"]
    )
print(dir())
