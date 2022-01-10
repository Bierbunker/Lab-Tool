# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.integrate import trapz
from scipy.integrate import simps
from sympy import sympify
from sympy.utilities.iterables import ordered
from sympy import latex
from sympy import Matrix, hessian, lambdify
import re
import math
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from sympy.interactive import printing
from math import sqrt
from math import floor
from pprint import pprint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas
import sympy
import scipy as sp
import numpy as np
__all__ = ["Equation"]


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
        print(list(variables.keys()))
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
        print(self)
        print(latex(self.err_expr))

    def error_func(self):
        vs = list(ordered(self.expr.free_symbols))
        print(self.expr.free_symbols)
        logging.info(vs)
        print(vs)

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0
        errs = " ".join([f"d{s}" for s in vs])
        er = sympy.symbols(errs)
        logging.info(er)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(self.expr, vs), er):
            # print(s)
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
        y = ((1 / (np.sqrt(2 * np.pi) * ser.std())) *
             np.exp(-0.5 * (1 / ser.std() * (bins - ser.mean())) ** 2))
        ax.plot(bins, y, '--')
        ax.set_xlabel(rf"${ser.name}$")
        ax.set_ylabel("$N$")
        ax.set_title(
            rf"Histogramm von ${name}$: $\mu={ser.mean()}$, $\sigma={round_up(ser.std(), 4)}$")
        fig.tight_layout()
        plt.savefig("histo.png")
        with open(f'../data/histo_data_{ser.name}.tex', 'w') as tf:
            tf.write(ser.rename(rf"${ser.name}$").to_latex(escape=False))

    def print_table(self, df):
        """Tries to export dataframe to latex table

        :df: TODO
        :returns: TODO

        """
        print(df.columns)
        colnames = list()
        for colname in df.columns:
            if "d" in colname and "\\" not in colname:
                colname = colname.replace("d", "\\Delta ")
            if len(colname) > 1 and "\\" not in colname:
                colname = "\\" + colname

            colname = "$" + colname + "$"
            colnames.append(colname)

        df.columns = colnames
        with open(f'../data/messreihe_{df.name}.tex', 'w') as tf:
            tf.write(df.to_latex(escape=False))

    def normalize(self):
        """converts none si units into si units
        :returns: TODO

        """
        pass

        """
        TODO write own to latex function export data with the use of siunix plugin
        """

    def objective(self, t, a, w, phi, offset):
        """Is the function used to fit after

        :A: TODO
        :w: TODO
        :phi: TODO
        :offset: TODO
        :returns: TODO

        """
        return a * np.sin(w * t + phi) + offset

    def plot_data(self, x, y, labels, show_lims=False, withfit=False, scaling=False):
        plt.cla()
        raw_x = self.data[x]
        raw_y = self.data[y]

        if scaling:
            plt.yscale(scaling)
            plt.xscale(scaling)
        else:
            plt.yscale('linear')
            plt.xscale('linear')
        x_data = np.linspace(raw_x.min(), raw_x.max(), 1000)
        plt.scatter(raw_x, raw_y, c="b", marker=".", label="Data")
        # plt.errorbar(raw_x, raw_y, yerr=self.data[f"d{y}"], fmt="none", capsize=3)

        if withfit:
            popt, pcov = curve_fit(self.objective, raw_x,
                                   raw_y, p0=[-70, 0.025, -1.5, 260])
            stdevs = np.sqrt(np.diag(pcov))
            print(popt)
            print(popt + stdevs)
            print(popt - stdevs)
            y_fit = self.objective(x_data, *popt)
            plt.plot(x_data, y_fit, c="r", label="Fit")
            if show_lims:
                plt.plot(x_data, self.objective(
                    x_data, *(popt + stdevs)), label="Obere Schranke")
                plt.plot(x_data, self.objective(
                    x_data, *(popt - stdevs)), label="Untere Schranke")
            chisq = chisq_stat(self.objective, self.data[x], self.data[y], popt, self.data[rf"d{y}"]) / len(
                self.data[x])
            deci = 2
            plt.annotate(rf"$\chi^{2}_{{dof}}$ = {round(chisq, deci)}", xy=(min(x_data), min(y_fit)),
                         bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.15 * min(y_fit)),
                         fontsize=13, arrowprops=dict(arrowstyle="-"))
            print(popt)
            print(pcov)
            print(stdevs)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.legend(loc=0)
        # for val, err in zip(popt,stdevs):
        #    plt.annotate(rf"$g = j{round(val,deci)} \pm {round_up(err,deci)}" + r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$" , xy=(min(x_data), min(y_fit)),
        #            bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.05 * min(y_fit)),
        #            fontsize=13, arrowprops=dict(arrowstyle="-"))

        ##chisq, pval = chisquare(self.data[y]/10,objective(self.data[x],*popt))

        # plt.annotate(rf"P-Value = {round(pval,deci)}" , xy=(min(x_data), min(y_fit)),
        #         bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.25 * min(y_fit)),
        #         fontsize=13, arrowprops=dict(arrowstyle="-"))
        # plt.show()
        plt.savefig(f"fit_of_{x}_{y}.png")
        # print(chisq(objective,self.data[x],self.data[y]/10,popt,self.data[f"d{y}"]/10))
        # print(chisq_stat(objective,self.data[x],self.data[y]/10,popt,self.data[rf"\sigma_{{{y}}}"]/10))

    def plot_linear_reg(self, x, y):
        """Calculates the linear regression of data given and plot it

        :x: TODO
        :y: TODO
        :returns: TODO

        """
        plt.cla()
        plt.scatter(x, y, c="b", marker=".", label="Data")
        plt.errorbar(x, y, yerr=self.data[f"d{y.name}"], fmt="none", capsize=3)
        plt.xlabel("Fadenlänge / m")
        plt.ylabel("Periodendauer zum Quadrat $T^{2}$/ $\\mathrm{s}^{2}$")
        X = x.to_numpy().reshape(-1, 1)
        Y = y.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(X, Y)
        x_fit = np.linspace(min(X), max(X), 100)
        y_fit = reg.predict(x_fit)
        popt, pcov = curve_fit(
            linear_func, x, y, p0=9, sigma=self.data[f"d{y.name}"] / 10 * 2, absolute_sigma=True)
        stdevs = np.sqrt(np.diag(pcov))
        deci = 2
        for val, err in zip(popt, stdevs):
            plt.annotate(
                rf"$g = {format(round(val, deci), f'.{deci}f')} \pm {round_up(err, deci)}" +
                r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$",
                xy=(min(x), min(y_fit)),
                bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x), 1.05 * min(y_fit)),
                fontsize=13, arrowprops=dict(arrowstyle="-"))
            plt.plot(x_fit, y_fit, c='r', label="Linearer Fit")
        plt.annotate("$R^{2}$:" + str(floor(reg.score(X, Y) * 10000) / 100) + "%", xy=(min(x_fit), min(y_fit)),
                     bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(1.25 * min(x_fit), 1.05 * min(y_fit)),
                     fontsize=13, arrowprops=dict(arrowstyle="-"))
        plt.legend(loc=0)
        plt.savefig(f"lin_reg_{x.name}_{y.name}.png")

    def plot_function(self):
        pass


def linear_func(x, g):
    """Plain old linear function

    :x: TODO
    :returns: TODO

    """
    return 4 * np.pi * np.pi * x / g


def t_factor_of(len):
    if len:
        return 1.06


# def chisq(f,x_data,y_data,popt,sigma):
#     """Calculates the chisq of fitted data

# :f: TODO
# :x_data: TODO
# :y_data: TODO
# :popt: TODO
# :sigma: TODO
# :returns: TODO

# """
# prediction = f(x_data, *popt)
# r = y_data - prediction
# chisq = np.sum((r)**2/prediction)
# return chisq

def chisq_stat(f, x_data, y_data, popt, sigma):
    prediction = f(x_data, *popt)
    r = y_data - prediction
    chisq = np.sum((r / sigma) ** 2)
    return chisq


# df = nobs - 2
# print(“chisq =”,chisq,”df =”,df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
