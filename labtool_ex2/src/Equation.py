# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy as sp
import sympy
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pprint import pprint
from math import floor
from math import sqrt
from sympy.interactive import printing
from scipy.optimize import curve_fit
from scipy.stats import chisquare

import math
import re

from sympy import Matrix, hessian, lambdify
from sympy import latex
from sympy.utilities.iterables import ordered
from sympy import sympify
from scipy.integrate import simps
from scipy.integrate import trapz

printing.init_printing(use_latex=True)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


class Equation:
    def __init__(self, expr, variables):
        self.variables = variables
        self.symbols = sympy.symbols(variables)
        self.expr = sympify(expr)
        self.err_expr = self.error_func()
        self.raw_data = "Data not loaded"
        self.data = pandas.DataFrame(data=None)
        self.sigfigs = dict()
        self.func = self.expr_to_np(self.expr)
        self.err_func = self.expr_to_np(self.err_expr)

    def error_func(self):
        vs = list(ordered(self.expr.free_symbols))
        def gradient(f, vs): return Matrix([f]).jacobian(vs)
        e_func = 0
        errs = ' '.join([f"d{s}" for s in vs])
        er = sympy.symbols(errs)
        for c, s in zip(gradient(self.expr, vs), er):
            e_func = e_func + abs(c) * s
        return e_func

    def load_data(self, path):
        df = pandas.read_csv(path, header=[0, 1], skipinitialspace=True)
        # df.set_labels(["type","variable"])
        df.columns = pandas.MultiIndex.from_tuples(
            df.columns, names=["type", "variable"])
        self.raw_data = df

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
            (np.sqrt(narray.var(ddof=1) / len(narray)) * t_factor_of(
                len(narray)) + linear_uncertainties) / multiplicity)
        return [res]

    def find_sigfigs(self, x):
        '''Returns the number of significant digits in a number. This takes into account
        strings formatted in 1.23e+3 format and even strings such as 123.450'''
        # change all the 'E' to 'e'
        x = x.lower()
        if ('e' in x):
            # return the length of the numbers before the 'e'
            myStr = x.split('e')
            return len(myStr[0]) - 1  # to compenstate for the decimal point
        else:
            # put it in e format and return the result of that
            # NOTE: because of the 8 below, it may do crazy things when it parses 9 sigfigs
            n = ('%.*e' % (8, float(x))).split('e')
            # remove and count the number of removed user added zeroes. (these are sig figs)
            if '.' in x:
                s = x.replace('.', '')
                # number of zeroes to add back in
                l = len(s) - len(s.rstrip('0'))
                # strip off the python added zeroes and add back in the ones the user added
                n[0] = n[0].rstrip('0') + ''.join(['0' for num in xrange(l)])
            else:
                # the user had no trailing zeroes so just strip them all
                n[0] = n[0].rstrip('0')
                # pass it back to the beginning to be parsed
        return find_sigfigs('e'.join(n))

    def columns_to_symbols(self, df, symbols):
        symbols = symbols.split(' ')
        df.columns = symbols
        print(df)

    def check(self):
        werte = self.raw_data['wert']
        vals = self.statistical_uncertainty(werte["T"]["TS1"].to_numpy(), linear_uncertainties=0.01,
                                            multiplicity=10)
        vals = pandas.DataFrame(vals, columns=["T", "dT"])
        print(werte[werte["l"]["Length"].astype(bool)])
        vals["l"] = werte[werte["l"]["Length"].astype(bool)]["l"]["Length"]
        print(vals)
        # self.columns_to_symbols(vals, 'T dT l')
        self.error_calculation(vals)

    def error_calculation(self, data):
        rows = data.to_dict("index")
        print(rows)
        for i, row in rows.items():
            print(row)
            row['dl'] = 0.003
            # val = self.func(**row)
            err = self.err_func(**row)
            # print(val)
            print(err)

    def resort(self):
        print(self.raw_data.columns.get_level_values('type'))
        for col, data in self.raw_data.items():
            print(col[1])

    def chunkify(self, criteria):

        pass

    def testing(self, use_min=True):
        """for testing
        :returns: TODO

        """
        for c in self.variables.split():
            reg_var = rf"^{c}(\.\d+)?$"
            reg_err = rf"^d{c}(\.\d+)?$"
            # a terrible solution probably should not allow second index
            #            try:
            #                index = self.raw_data.columns.droplevel('type')
            #            except:
            #                pass
            #            data = self.raw_data
            #            data.columns = index
            # end
            data = self.raw_data.droplevel("type", axis=1)
            filtered_vardata = data.filter(regex=reg_var)
            filtered_errdata = data.filter(regex=reg_err)
            # probably need to concat
            # self.raw_vardata[c] = filtered_vardata
            if len(filtered_vardata.columns) > 1:
                mean = filtered_vardata.mean()
                mean.name = c
                std = filtered_vardata.std()
                std = std.fillna(std.max())
                std.name = rf"\sigma_{{{c}}}"
                # i could check if a filtered error data is supplyied and
                # add it to the sem
                sem = filtered_vardata.sem()
                sem = sem.fillna(sem.max())
                sem.name = rf"d{c}"
                if use_min:
                    df = mean.to_frame()
                    df[std.name] = std.min()
                    df[sem.name] = sem.min()
                    df.reset_index(drop=True, inplace=True)
                else:
                    df = pandas.concat(
                        [mean, std, sem], axis=1).reset_index(drop=True)
                self.data = pandas.concat([self.data, df], axis=1)
            else:
                self.data = pandas.concat(
                    [self.data, filtered_vardata, filtered_errdata], axis=1)

        self.data.dropna(inplace=True)

    def get_vardata(self, raw_data=False):
        """Simply returns the data needed to calculate the equation
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():
            reg = rf"^{c}$"
            if raw_data:
                filt = self.raw_data.droplevel(
                    "type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    def get_errdata(self, raw_data=False):
        """Simply returns the error of the variables
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():
            reg = rf"^d{c}$"
            if raw_data:
                filt = self.raw_data.droplevel(
                    "type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    def get_vardata_and_errdata(self, raw_data=False):
        """Simply returns the variables with their errors
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():
            reg = rf"^(d)?{c}$"
            if raw_data:
                filt = self.raw_data.droplevel(
                    "type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    def var(self, var, raw_data=False):
        """Getter for a variable

        :var: TODO
        :returns: TODO

        """
        reg = rf"^(\\sigma_{{)?(d)?{var}(\.\d+)?(}})?$"
        if raw_data:
            df = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
        else:
            df = self.data.filter(regex=reg)
        df.name = var
        return df

    def get_extra_data(self, raw_data=True):
        """Getter for a extra data

        :returns: TODO

        """
        reg = rf"^!((\\sigma_{{)?(d)?{var}(\.\d+)?(}})?)(.)+$"
        if raw_data:
            df = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
        else:
            df = self.data.filter(regex=reg)
        df.name = "extra"
        return df

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
    I = trapz(P.data['p'], P.data['V']) / passes * hpacm3tosi
    print(I)
    c = 1
    for i in np.linspace(244, 2432, 10):
        print(i)
        df = P.data[(P.data.t < i)]
        I2 = trapz(df['p'], df['V']) / c * hpacm3tosi
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

#    #print(g.raw_data)
#    print(latex(g.expr))
#    print(latex(g.err_expr))
#    print(g.err_expr.free_symbols)
#    # g.check()
#    g.resort()
#    ser = g.raw_data.droplevel("type", axis=1)["T"]
#    g.testing()
#    print(g.var("T",raw_data=True))
#    g.histoofseries(ser,5,"Periodauern")
#    print(g.data)
#    print(g.get_vardata())
#    g.data["l"] = g.data["l"] + 0.057
#    g.plot_data("l","T")
#    x = g.data["l"]
#    y = (g.data["T"]/10)**2
#    g.plot_linear_reg(x,y)
#    g.print_table(g.var("T"))
#    g.print_table(g.var("l"))
#    winkeln = pandas.concat([g.var("l"),g.var("x", raw_data=True),g.var("phi", raw_data=True)],axis=1)
#    winkeln.name = "winkel"
#    winkeln.dropna(inplace=True)
#    g.print_table(winkeln)
#    g.find_sigfigs("0.0050")
#    g.find_sigfigs("0.005")
#    g.find_sigfigs("0.205")
#
