# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import sympy
import math
import re
import copy
import matplotlib
import uncertainties
import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas
from sympy.interactive import printing
from lmfit.models import ExpressionModel
from collections.abc import Iterable


from sympy import Matrix, hessian, lambdify
from sympy import latex
from sympy.utilities.iterables import ordered
from sympy import sympify
from sympy import Eq
from sympy.printing.latex import latex
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)

transformations = standard_transformations + (implicit_multiplication_application,)
from scipy.integrate import trapz

printing.init_printing(use_latex=True)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


class Equation:
    def __init__(
        self,
        expr,
        variables,
        project_name,
        label,
        var_name,
        mapping=dict(),
        add_subplot=False,
        dataframe=None,
        figure=None,
    ):
        # variables if you don't have data on it dont pass it
        # variables = {
        # "f":"mm"
        # "r":"mm"
        # "NA":"1"
        # }
        # settings
        self.var_name = var_name
        self.variables = variables
        self.project_name = project_name
        self.label = label
        self.mapping = mapping
        self.add_subplot = add_subplot

        # sympy variable stuff
        self.symbols = sympy.symbols(list(variables.keys()))
        self.var_name_symbol = sympy.symbols(var_name)

        # sympy eq stuff
        # if errors are encountered it is probably a naming confliction with internal functions like rf see https://docs.sympy.org/latest/modules/functions/combinatorial.html
        self.expr = sympify(expr)
        self._expr = expr
        self.eqn = Eq(self.var_name_symbol, self.expr)
        self.err_expr = self.error_func()
        # self.expr = parse_expr(expr, transformations=transformations)

        # data
        self.data = dataframe
        # self.sigfigs = dict()

        # internal numpy representation of the Equation for fast calculation
        self.func = self.expr_to_np(self.expr)
        self.err_func = self.expr_to_np(self.err_expr)

        # plotting
        if figure:
            self.figure = figure
        else:
            self.figure = plt.figure()

    def error_func(self):
        vs = list(ordered(self.expr.free_symbols))
        logging.info(vs)

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0
        errs = " ".join([f"d{s}" for s in vs])
        er = sympy.symbols(errs)
        logging.info(er)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(self.expr, vs), er):
            e_func = e_func + abs(c) * s
        return e_func

    def expr_to_np(self, esp):
        return lambdify(tuple(esp.free_symbols), esp, "numpy")

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def __str__(self):
        tex = str(latex(self.eqn))
        # Nothing really working to get it to work LatexPrinter needs to be modified
        # https://stackoverflow.com/questions/43350381/sympy-modifying-latex-output-of-derivatives
        # for var in self.eqn.free_symbols:
        #     var = str(var)
        #     if var in self.mapping:
        #         print(var)
        #         tex.replace(var, self.mapping[var])
        #         expr.subs()
        # return latex(self.expr)
        return tex

    def __doc__(self):
        return """
        This is the incomplete Documentation of Equation
        If something doesn't work ask Max he knows and he will tell you it
        will be fixed some time and you should make an issue.

        """

    # @multiplicity for example if data is of ten period instead of one
    # TODO Andi
    # def statistical_uncertainty(self, narray, linear_uncertainties=0.0, multiplicity=1):
    #     res = list()
    #     res.append(narray.mean() / multiplicity)
    #     res.append(
    #         (
    #             np.sqrt(narray.var(ddof=1) / len(narray)) * t_factor_of(len(narray))
    #             + linear_uncertainties
    #         )
    #         / multiplicity
    #     )
    #     return [res]

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

    def _solve_for(self, var):
        eqn = solve(self.eqn, sympy.symbols(var))
        if isinstance(eqn, Iterable):
            eqn = eqn[1]
        return eqn

    def solve_for(self, var):
        # TODO should use internal _solve_for to solve for var and create a new Equation and return it
        eqn = self._solve_for(var)
        creation_params = {
            "expr": eqn,
            "variables": self.variables,
            "project_name": self.project_name,
            "label": "Change the label before plotting",
            "var_name": var,
            "mapping": self.mapping,
            "add_subplot": self.add_subplot,
            "dataframe": self.data,
            "figure": self.figure,
        }
        return Equation(**creation_params)

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

    def get_fig_ax(self, figure, toggle_add_subplot):
        fig = None
        ax = None
        if figure:
            fig = figure
            print(fig.get_axes())
            if toggle_add_subplot:
                if self.add_subplot:
                    if fig.get_axes():
                        ax = fig.get_axes()[0]
                    else:
                        ax = fig.add_subplot()
                else:
                    ax = fig.add_subplot()
            else:
                if self.add_subplot:
                    ax = fig.add_subplot()
                else:
                    if fig.get_axes():
                        ax = fig.get_axes()[0]
                    else:
                        ax = fig.add_subplot()
            # ax = fig.add_axes(
            #     rect=[x_data.min(), y_data.min(), x_data.max(), y_data.max()]
            # )
        elif self.figure:
            fig = self.figure
            print(fig.get_axes())
            if toggle_add_subplot:
                if self.add_subplot:
                    print(fig.get_axes())
                    if fig.get_axes():
                        ax = fig.get_axes()[0]
                    else:
                        ax = fig.add_subplot()
                else:
                    ax = fig.add_subplot()
            else:
                if self.add_subplot:
                    ax = fig.add_subplot()
                else:
                    if fig.get_axes():
                        ax = fig.get_axes()[0]
                    else:
                        ax = fig.add_subplot()
        else:
            raise Exception
        return fig, ax

    def apply_df(self):
        function_data = dict()
        # for i in range(len(self.data.columns)):
        #     series = self.data.iloc[:, i]
        #     var = series.name
        #     function_data[var] = series.to_numpy()
        # maybe better
        for var in self.expr.free_symbols:
            var = str(var)
            series = self.data[var]
            function_data[var] = series.to_numpy()
        return self(**function_data)

    def apply_df_err(self):
        function_data = dict()
        for var in self.err_expr.free_symbols:
            var = str(var)
            series = self.data[var]
            function_data[var] = series.to_numpy()
        return self.err_func(**function_data)

    def plot_data_scatter(
        self,
        x,
        figure=None,
        style="r",
        errors=False,
        withfit=False,
        guess=None,
        show_lims=False,
        scaling=False,
        toggle_add_subplot=False,
        **kwargs,
    ):
        # plt.cla()
        fig, ax = self.get_fig_ax(figure, toggle_add_subplot)
        x_data = self.data[x]
        y_data = self.apply_df()

        if scaling:
            ax.set_yscale(scaling)
            ax.set_xscale(scaling)
        else:
            ax.set_yscale("linear")
            ax.set_xscale("linear")
        ax.scatter(
            x_data,
            y_data,
            c=style,
            marker=".",
            s=39.0,
            label="Data",
        )
        if errors:
            y_err = self.apply_df_err()
            print(y_err)
            ax.errorbar(
                x_data,
                y_data,
                xerr=self.data[f"d{x}"],
                yerr=y_err,
                fmt="none",
                capsize=3,
            )

        if withfit:
            self.plot_fit(
                x=x, y=self.var_name, guess=guess, toggle_add_subplot=toggle_add_subplot
            )

        self.set_x_y_label(ax=ax, x=x, y=self.var_name)

        # ax.set_xlabel(labels[0])
        # ax.set_ylabel(labels[1])
        ax.legend(loc=0)
        # fig.savefig(
        #     f"./Output/{self.project_name}/{x}_{self.var_name}_mess_daten.png",
        #     dpi=400,
        # )
        # fig.savefig(
        #     f"./Output/{self.name}/fit_of_{x}_{y}_mess_nr_{self.load_data_counter}.png",
        #     dpi=400,
        # )

    def try_generate_missing_data(self):
        # takes a variable and
        pass

    def plot_fit(
        self,
        x,
        y,
        style="r-",
        use_all_known=False,
        offset=[0, 0],
        figure=None,
        guess=None,
        toggle_add_subplot=False,
    ):
        eqn = self._solve_for(y)
        print(eqn)
        print(eqn.free_symbols)
        try:
            x_data = self.data[x]
        except KeyError as e:
            print(f"No Data found for {x}")
            raise Exception
        try:
            y_data = self.data[y]
        except KeyError as e:
            print(f"No Data found for {y} trying to generate it")
            # TODO need to improved maybe with a solve_for y and then calculation
            y_data = pandas.Series(self.apply_df(), name=self.var_name)
        fig, ax = self.get_fig_ax(figure, toggle_add_subplot)
        if use_all_known:
            mod = ExpressionModel(
                str(eqn),
                independent_vars=[
                    str(sym) for sym in eqn.free_symbols if str(sym) == y
                ],
            )
        else:
            mod = ExpressionModel(
                str(eqn),
                independent_vars=[x],
            )

        # pars = mod.guess(raw_y, x=raw_x)
        if guess:
            pars = mod.make_params(**guess)
        else:
            # TODO custom error needs to be implemented
            print("No guess was passed")
            raise Exception
        # thinnen the fit data to avoid over fitting
        # x_fit = np.array_split(x_data.iloc[::2], 2)[0]
        # y_fit = np.array_split(y_data.iloc[::2], 2)[0]
        x_fit = x_data.values.reshape(1, -1)
        y_fit = y_data.values.reshape(1, -1)
        stuff = {x_data.name: x_fit}
        out = mod.fit(y_fit, pars, **stuff)
        x_continues_plot_data = np.linspace(x_data.min(), x_data.max(), 1000)
        for i, (name, param) in enumerate(out.params.items()):
            print("{:7s} {:11.5f} {:11.5f}".format(name, param.value, param.stderr))
            deci = -orderOfMagnitude(param.stderr)
            print(deci)
            if deci < 0:
                deci = 1
            ax.text(
                s=rf"${name} = {format(round(param.value, deci), f'.{deci}f')} \pm {round_up(param.stderr, deci)}$ "
                + self.variables[name],
                bbox={"facecolor": "#616161", "alpha": 0.85},
                x=(max(x_data) - min(y_data)) / 100 * (5 + offset[0]) + min(x_data),
                y=(max(y_data) - min(y_data)) * (offset[1] / 100 + (i + 1) / 7)
                + min(y_data),
                fontsize=10,
            )
        stuff2 = {x_data.name: x_continues_plot_data}
        resp = np.array(out.eval(**stuff2))[0, :]
        ax.plot(x_continues_plot_data, resp, style, label=f"{self.label} fit")
        self.set_x_y_label(ax, x, y)
        ax.legend(loc=0)
        print(out.fit_report(min_correl=0.25))
        fig.savefig(f"./Output/{self.project_name}/fit_of_{x}_to_{y}.png")

    def set_x_y_label(self, ax, x, y):
        unitx = self.variables[x]
        unity = self.variables[y]
        if x in self.mapping:
            x = self.mapping[x]
        if y in self.mapping:
            y = self.mapping[y]
        xlabel = rf"${x}$ / {unitx}"
        ylabel = rf"${y}$ / {unity}"
        if ax.get_xlabel() != xlabel:
            axtra = ax.twinx()
            axtra.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel)

        if ax.get_ylabel() != ylabel:
            axtra = ax.twiny()
            axtra.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel)

    def plot_function(self, x, style="r-", figure=None, toggle_add_subplot=None):
        fig, ax = self.get_fig_ax(figure, toggle_add_subplot)
        x_data = self.data[x]
        x_continues_plot_data = np.linspace(x_data.min(), x_data.max(), 1000)
        function_data = dict()
        print(str(self.eqn.free_symbols))
        for var in self.eqn.free_symbols:
            var = str(var)
            if var is not x and var is not self.var_name:
                series = self.data[var]
                function_data[var] = series.to_numpy()
        function_data[x] = x_continues_plot_data
        y_data = self(**function_data)
        self.set_x_y_label(ax, x, self.var_name)
        ax.plot(
            x_continues_plot_data,
            y_data,
            style,
            label=f"{self.label} Graph",
        )
        ax.legend(loc=0)
        # fig.savefig(f"./Output/{self.project_name}/plot_{x}_to_{self.var_name}.png")


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
