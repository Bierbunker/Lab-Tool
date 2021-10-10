# import labtool_ex2 as lt

from labtool_ex2 import Equation
from labtool_ex2 import Project

if __name__ == "__main__":
    P = Project("testing", font=13)
    P.load_data("./src/labtool_ex2/Data/abbe/nadatablue")
    print(P.data)
    na = Equation(
        "r / f", "r f",P.name, label="Numerische Apertur", dataframe=P.data[["r", "f"]]
    )
    P.equations["NA"] = na
    na.plot_data_scatter("f", labels=["bro", "test"])
print(dir())
