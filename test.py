# import labtool_ex2 as lt

from labtool_ex2 import Equation
from labtool_ex2 import Project

if __name__ == "__main__":
    P = Project("testing", font=13)
    P.load_data("./src/labtool_ex2/Data/abbe/nadatablue")
    print(P.data)
    na = Equation(
        "r / f", "r f", P.name, label="Numerische Apertur", dataframe=P.data[["r", "f", 'dr','df']]
    )
    print(na.err_expr)
    P.data["NA"] = na.apply_df()
    P.data["dNA"] = na.apply_df_err()
    P.equations["NA"] = na
    na.plot_data_scatter("r", errors=True, labels=["r / mm", "N.A. / 1"])
print(dir())
