# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from Project import Project
from Equation import Equation


def make_project_stuff(project, start, end):
    zeros = project.probe_for_zeros(start, end, "wx")
    period = (zeros.iloc[-1] - zeros.iloc[0]) / (len(zeros) // 2)
    project.local_ledger["period"].append(period)
    print("\n\n**************************************************************")
    print(f"Periode {period}")
    print("**************************************************************\n\n")
    project.data = project.data - (project.data["t"].iloc[0], 0, 0, 0)
    project.working_dfs.append(project.data.reset_index(drop=True))
    project.plot_data(
        "t", "wx", labels=["Zeit / s", r"$\omega_x$ / rad s$^{-1}$"], withfit=True
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    P = Project("testing", font=13)
    P.load_data("./Data/abbe/nadatablue")
    print(P.data)
    na = Equation("r / f", "r f", label="Numerische Apertur", data=P.data["r", "f"])
    P.equations["NA"] = na
    na.plot_data_scatter("f", labels=["bro", "test"])

#     P.local_ledger["period"] = list()

#     P.load_data("Data/Reibung/gleit.csv")

#     P.data["t"] = P.data["t"] / P.data["t"].mean()
#     fig, ax = plt.subplots()
#     category = ["Rutschzeit"] * len(P.data['t'])
#     ax.scatter(P.data["t"], category, c="b", label="Messwerte")
#     ax.scatter(P.data["t"].mean(), category[0], c="r", label="Mittelwert")
#     ax.errorbar(P.data["t"].mean(), [category[0]], c="r", capsize=3, xerr=P.data["t"].sem())
#     fig.suptitle("Relative Darstellung der Rutschzeitmesswerte")
#     ax.legend()

#     plt.savefig("Rutschzeitmesswerte.png", dpi=400)

#     P.load_data("Data/Reibung/gleit.csv", loadnew=True, clean=False)
#     P.data = P.raw_data["wert"]
#     print(P.data)
#     plt.cla()
#     P.data["ft"] = P.data["ft"] / P.data["ft"].mean()
#     print(P.data["ft"])
#     fig, ax = plt.subplots()
#     category = ["Fallzeit"] * len(P.data['ft'])
#     ax.scatter(P.data["ft"], category, c="b", label="Messwerte")
#     ax.scatter(P.data["ft"].mean(), category[0], c="r", label="Mittelwert")
#     ax.errorbar(P.data["ft"].mean(), [category[0]], c="r", capsize=3, xerr=P.data["ft"].sem())
#     fig.suptitle("Relative Darstellung der Fallzeitmesswerte")
#     ax.legend()

#     plt.savefig("Fallzeitmesswerte.png", dpi=400)

#     P.load_data("Data/Reibung/ergebnisse.csv", loadnew=True)

#     i = 0
#     categories = [r"$\mu_H$", r"$\mu_G$", r"$\mu_R$", "$\eta$"]
#     plt.cla()
#     fig, ax = plt.subplots()
#     for col in P.data.columns:
#         if "d" in col:
#             continue
#         err = P.data["d" + col] / P.data[col]
#         wert = P.data[col] / P.data[col]
#         ax.scatter(wert, categories[i])
#         ax.errorbar(wert, [categories[i]], capsize=3, xerr=err)
#         fig.suptitle("Relative Unsicherheit der erhaltenen Werte")
#         i += 1

#     plt.savefig("ergebnisse.png", dpi=400)

#     # P.plot_data("n", "t", labels=["1", "Zeit"])
