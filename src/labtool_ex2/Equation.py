# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy.interactive import printing
from lmfit.models import ExpressionModel

import math
import re

from sympy import Matrix, hessian, lambdify
from sympy import latex
from sympy.utilities.iterables import ordered
from sympy import sympify
from scipy.integrate import trapz

printing.init_printing(use_latex=True)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


class Equation:
    def __init__(self, expr, variables,project_name, label, dataframe=None, figure=None):
        self.variables = variables
        self.symbols = sympy.symbols(variables)
        self._expr = expr
        self.expr = sympify(expr)
        self.err_expr = self.error_func()
        self.project_name = project_name
        # self.data = pandas.DataFrame(data=None)
        self.data = dataframe
        # self.sigfigs = dict()
        self.func = self.expr_to_np(self.expr)
        self.err_func = self.expr_to_np(self.err_expr)
        self.figure = plt.figure()

    def error_func(self):
        vs = list(ordered(self.expr.free_symbols))

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0
        errs = " ".join([f"d{s}" for s in vs])
        er = sympy.symbols(errs)
        for c, s in zip(gradient(self.expr, vs), er):
            e_func = e_func + abs(c) * s
        return e_func

    def expr_to_np(self, esp):
        for x in range(10):
            print(tuple(esp.free_symbols))
        return lambdify(tuple(esp.free_symbols), esp, "numpy")

    def __call__(self, *args, **kwargs):
        return self.func(**kwargs)

    # @multiplicity for example if data is of ten period instead of one
    def statistical_uncertainty(self, narray, linear_uncertainties=0.0, multiplicity=1):
        res = list()
        res.append(narray.mean() / multiplicity)
        res.append(
            (
                np.sqrt(narray.var(ddof=1) / len(narray)) * t_factor_of(len(narray))
                + linear_uncertainties
            )
            / multiplicity
        )
        return [res]

    def find_sigfigs(self, x):
        """Returns the number of significant digits in a number. This takes into account
        strings formatted in 1.23e+3 format and even strings such as 123.450"""
        # change all the 'E' to 'e'
        x = x.lower()
        if "e" in x:
            # return the length of the numbers before the 'e'
            myStr = x.split("e")
            return len(myStr[0]) - 1  # to compenstate for the decimal point
        else:
            # put it in e format and return the result of that
            # NOTE: because of the 8 below, it may do crazy things when it parses 9 sigfigs
            n = ("%.*e" % (8, float(x))).split("e")
            # remove and count the number of removed user added zeroes. (these are sig figs)
            if "." in x:
                s = x.replace(".", "")
                # number of zeroes to add back in
                l = len(s) - len(s.rstrip("0"))
                # strip off the python added zeroes and add back in the ones the user added
                n[0] = n[0].rstrip("0") + "".join(["0" for num in xrange(l)])
            else:
                # the user had no trailing zeroes so just strip them all
                n[0] = n[0].rstrip("0")
                # pass it back to the beginning to be parsed
        return find_sigfigs("e".join(n))

    def columns_to_symbols(self, df, symbols):
        symbols = symbols.split(" ")
        df.columns = symbols
        print(df)

    def error_calculation(self, data):
        rows = data.to_dict("index")
        print(rows)
        for i, row in rows.items():
            print(row)
            # val = self.func(**row)
            err = self.err_func(**row)
            # print(val)
            print(err)

    # def resort(self):
    #     print(self.raw_data.columns.get_level_values("type"))
    #     for col, data in self.raw_data.items():
    #         print(col[1])

    def histoofseries(self, ser, bins, name):
        """Plots the histogram of a pandas series and draws a fit

        :ser: TODO
        :bins: TODO
        :name: TODO
        :returns: TODO

        """
        fig, ax = plt.subplots()
        bins = np.linspace(ser.min(), ser.max(), bins + 1)
        ser.hist(ax=ax, bins=bins)
        y = (1 / (np.sqrt(2 * np.pi) * ser.std())) * np.exp(
            -0.5 * (1 / ser.std() * (bins - ser.mean())) ** 2
        )
        ax.plot(bins, y, "--")
        ax.set_xlabel(rf"${ser.name}$")
        ax.set_ylabel("$N$")
        ax.set_title(
            rf"Histogramm von ${name}$: $\mu={ser.mean()}$, $\sigma={round_up(ser.std(), 4)}$"
        )
        fig.tight_layout()
        plt.savefig("histo.png")
        with open(f"../data/histo_data_{ser.name}.tex", "w") as tf:
            tf.write(ser.rename(rf"${ser.name}$").to_latex(escape=False))

    def plot_data_scatter(
        self,
        x,
        labels,
        guess=None,
        figure=None,
        errors=False,
        show_lims=False,
        withfit=False,
        scaling=False,
    ):
        # plt.cla()
        fig = None
        ax = None
        function_data = dict()
        for i in range(len(self.data.columns)):
            series = self.data.iloc[:, i]
            var = series.name
            function_data[var] = series.to_numpy()

        # maybe better
        # for var in self.variables.split("\\s+"):
        #     series = self.df[var]
        #     function_data[var] = series.to_numpy()

        x_data = self.data[x]
        y_data = self(**function_data)
        print(x_data)
        print(y_data)
        if figure:
            fig = figure
            ax = fig.add_axes(
                rect=[x_data.min(), y_data.min(), x_data.max(), y_data.max()]
            )
        elif self.figure:
            fig = self.figure
            ax = fig.add_axes(
                rect=[x_data.min(), y_data.min(), x_data.max(), y_data.max()]
            )
        else:
            raise Exception
        x_continues_plot_data = np.linspace(x_data.min(), x_data.max(), 1000)

        if scaling:
            ax.set_yscale(scaling)
            ax.set_xscale(scaling)
        else:
            ax.set_yscale("linear")
            ax.set_xscale("linear")
        ax.scatter(
            x_data,
            y_data,
            c="r",
            marker=".",
            s=39.0,
            label="Data",
        )
        if errors:
            ax.errorbar(
                raw_x,
                raw_y,
                xerr=self.data[f"d{x}"],
                yerr=self.data[f"d{y}"],
                fmt="none",
                capsize=3,
            )

        if withfit:
            print(self.variables.split("\\s+"))
            mod = ExpressionModel(
                self._expr,
                independent_vars=self.variables.split("\\s+"),
            )
            # pars = mod.guess(raw_y, x=raw_x)
            if guess:
                pars = mod.make_params(**guess)
            else:
                raise Exception
            # thinnen the fit data to avoid over fitting
            x_fit = np.array_split(raw_x.iloc[::2], 2)[0]
            y_fit = np.array_split(raw_y.iloc[::2], 2)[0]
            out = mod.fit(y_fit, pars, x=x_fit)
            ax.plot(x_fit, out.init_fit, "b-", label="fit" + self.label)
            # ax.plot(raw_x[:len(raw_x)//2], (10 * np.exp(-0.157 * raw_x[:len(raw_x)//2])), "m-", label="exponentiell")
            # ax.plot(raw_x[len(raw_x)//2:], (-0.05 * raw_x[len(raw_x)//2:] + 2.4), "g-", label="linear")
            ax.plot(raw_x, out.eval(x=raw_x), "r-", label="best fit")
            print(out.fit_report(min_correl=0.25))

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend(loc=0)
        fig.savefig(
            f"./Output/{self.project_name}/fit_of_{x}__mess_nr_.png",
            dpi=400,
        )
        # fig.savefig(
        #     f"./Output/{self.name}/fit_of_{x}_{y}_mess_nr_{self.load_data_counter}.png",
        #     dpi=400,
        # )

    def plot_function(self):
        pass


if __name__ == "__main__":
    # expr = f"4 * {np.pi * np.pi} / (T * T) * l"
    expr = f"p*V/t * Vf * f"

    P = Equation(expr, "V p t Vf f")
    # g.load_data("../data/rawdata.csv")
    P.load_data("unbelastet.csv")
    P.data = P.data.astype(float)
    P.testing()
    P.data["dV"] = 5.00
    P.data["dp"] = 1.00
    print(P.data)
    P.plot_data("t", "V", labels=["V / cm$^3$", "p / hPa"])
    P.plot_data("V", "p", labels=["V / cm$^3$", "p / hPa"])
    passes = 10
    hpacm3tosi = 1 / 10000
    # df = P.data[(P.data.t < 244)]
    # df = P.data[(P.data.t < 2432)]
    print("Integrals")
    I = trapz(P.data["p"], P.data["V"]) / passes * hpacm3tosi
    print(I)
    c = 1
    for i in np.linspace(244, 2432, 10):
        print(i)
        df = P.data[(P.data.t < i)]
        I2 = trapz(df["p"], df["V"]) / c * hpacm3tosi
        c += 1
        print(I2)
    print("End Integrals")
    drehzahl = 1 / (2432 / 10000)
    print(f"Drehzahl aus Daten {drehzahl}")
    print(f"Mechanische Leistung {drehzahl * 1.87}")
    print(f"Heizleistung Leistung 107.5 W")
    print(f"Wirkungsgrad {drehzahl * 1.87 / 107.5}")

    P.data = P.data[(2 < P.data.f) & (P.data.f < 7)]
    P.plot_data("f", "Vf", labels=["f / s$-1$", "V(f) / cm$^3$"])
