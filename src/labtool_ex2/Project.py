# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

# import scipy as sp
# import sympy
import pandas
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Parameters

# from math import floor
# from math import sqrt
from sympy.interactive import printing
from lmfit.models import ExpressionModel
from lmfit.models import LinearModel

from pathlib import Path

import math
import re

from sympy import Matrix, hessian, lambdify
from sympy import latex

from .Equation import Equation  # relative import

printing.init_printing(use_latex=True)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


class Project:
    def __init__(self, name, font=10):
        # self.variables = variables
        # self.symbols = sympy.symbols(variables)
        self.name = name
        self.equations = dict()
        self.load_data_counter = 0
        # self.expr = sympify(expr)
        # self.err_expr = self.error_func()
        self.data_path = list()
        self.messreihen_dfs = list()
        self.working_dfs = list()
        self.local_ledger = dict()
        self.raw_data = pandas.DataFrame(data=None)
        self.data = pandas.DataFrame(data=None)
        self.dfs = dict()
        plt.rcParams.update({"font.size": font})
        plt.rcParams.update({"xtick.labelsize": 9})
        plt.rcParams.update({"ytick.labelsize": 9})

        p = Path(f"./Output/{name}/")
        p.mkdir(parents=True, exist_ok=True)

    def load_data(self, path, loadnew=False, clean=True):
        print("\n\n\nLoading Data from: " + path)
        if loadnew:
            self.data = pandas.DataFrame(data=None)
        df = pandas.read_csv(path, header=[0, 1], skipinitialspace=True)
        df.columns = pandas.MultiIndex.from_tuples(
            df.columns, names=["type", "variable"]
        )
        self.load_data_counter += 1
        self.raw_data = df.astype(float)
        if clean:
            name = Path(path).stem
            self.clean_dataset(name=name)

    def clean_dataset(self, name, use_min=False):
        for var in self.raw_data.droplevel("type", axis=1).columns:
            if not re.match(r"^d(\w)+(\.\d+)?$", var):
                reg_var = rf"^{var}(\.\d+)?$"
                reg_err = rf"^d{var}(\.\d+)?$"
                data = self.raw_data.droplevel("type", axis=1)
                filtered_vardata = data.filter(regex=reg_var)
                filtered_errdata = data.filter(regex=reg_err)
                # probably need to concat
                # self.raw_vardata[c] = filtered_vardata
                if len(filtered_vardata.columns) > 1:
                    mean = filtered_vardata.mean()
                    mean.name = var
                    std = filtered_vardata.std()
                    std = std.fillna(std.max())
                    std.name = rf"\sigma_{{{var}}}"
                    # i could check if a filtered error data is supplied and
                    # add it to the sem
                    sem = filtered_vardata.sem()
                    sem = sem.fillna(sem.max())
                    sem.name = rf"d{var}"
                    if use_min:
                        df = mean.to_frame()
                        df[std.name] = std.min()
                        df[sem.name] = sem.min()
                        df.reset_index(drop=True, inplace=True)
                    else:
                        df = pandas.concat([mean, std, sem], axis=1).reset_index(
                            drop=True
                        )
                    self.data = pandas.concat([self.data, df], axis=1)
                else:
                    self.data = pandas.concat(
                        [self.data, filtered_vardata, filtered_errdata], axis=1
                    )
            else:
                continue

        self.data.dropna(inplace=True)
        self.messreihen_dfs.append(self.data)
        self.dfs[name] = self.data

    def find_possible_zero(self, identifier):
        return self.data[~self.data[identifier].astype(bool)]

    def normalize(self):
        """converts none si units into si units
        :returns: TODO

        """
        pass

    def get_vardata(self, raw_data=False):
        """Simply returns the data needed to calculate the equation
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():
            reg = rf"^{c}$"
            if raw_data:
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
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
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
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
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
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

    def get_extra_data(self, var, raw_data=True):
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
        with open(
            f"../data/histo_data_{ser.name}_mess_nr_{self.load_data_counter}", "w"
        ) as tf:
            tf.write(ser.rename(rf"${ser.name}$").to_latex(escape=False))

    """
    TODO write own to latex function export data with the use of siunitx plugin
    """

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
        with open(f"../data/messreihe_{df.name}.tex", "w") as tf:
            tf.write(df.to_latex(escape=False))

    def objective(self, beta, t):
        """Is the function used to fit after

        :A: TODO
        :w: TODO
        :phi: TODO
        :offset: TODO
        :returns: TODO

        """
        return -beta[0] * np.sin(beta[1] * t) * np.exp(beta[2] * t)

    def plot_data(self, x, y, labels, show_lims=False, withfit=False, scaling=False):
        plt.cla()
        raw_x = self.data[x]
        raw_y = self.data[y]

        if scaling:
            plt.yscale(scaling)
            plt.xscale(scaling)
        else:
            plt.yscale("linear")
            plt.xscale("linear")
        x_data = np.linspace(raw_x.min(), raw_x.max(), 1000)
        # length = len(raw_x)
        # middle_index = length // 2

        # plt.scatter(
        #     raw_x[:middle_index],
        #     raw_y[:middle_index],
        #     c="b",
        #     marker=".",
        #     s=39.0,
        #     label="Blau Data",
        # )
        # plt.scatter(
        #     raw_x[middle_index:],
        #     raw_y[middle_index:],
        #     c="r",
        #     marker=".",
        #     s=39.0,
        #     label="Rot Data",
        # )
        plt.scatter(
            raw_x,
            raw_y,
            c="r",
            marker=".",
            s=39.0,
            label="Rot Data",
        )
        if errors:
            plt.errorbar(
                raw_x,
                raw_y,
                xerr=self.data[f"d{x}"],
                yerr=self.data[f"d{y}"],
                fmt="none",
                capsize=3,
            )

        # y_blau_fit = 0.61 * 470 * 10 ** (-6) / x_data
        # y_red_fit = 0.61 * 635 * 10 ** (-6) / x_data
        # plt.plot(x_data, y_blau_fit, "b-", label="Blau Theoretisch")
        # # plt.plot(x_data, y_blau_fit, "b-", label="Blau Theoretisch")
        # plt.plot(x_data, y_red_fit, "r-", label="Red Theoretisch")

        if withfit:
            # mod = ExpressionModel(
            #     "-amp * exp(-x/l) * (cos(x*phase + shift)/l + phase * sin(x*phase + shift))",
            #     independent_vars=["x"],
            # )
            mod = ExpressionModel(
                "0.61 * l / NA",
                independent_vars=["x"],
            )
            # pars = mod.guess(raw_y, x=raw_x)
            pars = mod.make_params(amp=1.5, l=20, phase=7, shift=0.1)
            # print(raw_y.iloc[::2])
            # print(raw_x.iloc[::2])
            # x_fit = raw_x
            # y_fit = raw_y
            # x_fit = raw_x.iloc[::2].to_numpy()
            # y_fit = raw_y.iloc[::2].to_numpy()
            # x_fit = np.concatenate(np.array_split(raw_x.iloc[::2], 4)[::2])
            # y_fit = np.concatenate(np.array_split(raw_y.iloc[::2], 4)[::2])
            x_fit = np.array_split(raw_x.iloc[::2], 2)[0]
            y_fit = np.array_split(raw_y.iloc[::2], 2)[0]
            # print(len(x_fit))
            # print(len(y_fit))
            out = mod.fit(y_fit, pars, x=x_fit)
            # plt.plot(x_fit, out.init_fit, 'b-', label='initial fit')
            # plt.plot(raw_x[:len(raw_x)//2], (10 * np.exp(-0.157 * raw_x[:len(raw_x)//2])), "m-", label="exponentiell")
            # plt.plot(raw_x[len(raw_x)//2:], (-0.05 * raw_x[len(raw_x)//2:] + 2.4), "g-", label="linear")
            # plt.plot(raw_x, out.eval(x=raw_x), 'r-', label='best fit')
            print(out.fit_report(min_correl=0.25))
            # x_fit = np.linspace(min(raw_x), max(raw_x), 100)
            # data = RealData(raw_x, raw_y, self.data[f"d{x}"], self.data[f"d{y}"])
            # data = RealData(np.array_split(raw_x.iloc[::2], 7)[5], np.array_split(raw_y.iloc[::2], 7)[5])
            # model = Model(self.objective)
            #
            # odr = ODR(data, model, [10, 8, -0.2])
            # odr.set_job(fit_type=2)
            # output = odr.run()
            # print("Section using ODR")
            # print(output.beta)
            # print(output.sd_beta)
            # plt.plot(x_fit, self.objective(output.beta, x_fit), label="Obere Schranke")
            # if show_lims:
            #     plt.plot(x_fit, self.objective(output.beta + output.sd_beta, x_fit), label="Obere Schranke")
            #     plt.plot(x_fit, self.objective(output.beta - output.sd_beta, x_fit), label="Untere Schranke")

            # popt, pcov = curve_fit(self.objective, raw_x, raw_y, p0=[-70, 0.025, -1.5, 260])
            # stdevs = np.sqrt(np.diag(pcov))
            # print(popt)
            # print(popt + stdevs)
            # print(popt - stdevs)
            # y_fit = self.objective(x_data, *popt)
            # plt.plot(x_data, y_fit, c="r", label="Fit")
            # if show_lims:
            #     plt.plot(x_data, self.objective(x_data, *(popt + stdevs)), label="Obere Schranke")
            #     plt.plot(x_data, self.objective(x_data, *(popt - stdevs)), label="Untere Schranke")
            # chisq = self.chisq_stat(self.objective, self.data[x], self.data[y], popt, self.data[rf"d{y}"]) / len(
            #     self.data[x])
            # deci = 2
            # plt.annotate(rf"$\chi^{2}_{{dof}}$ = {round(chisq, deci)}", xy=(min(x_data), min(y_fit)),
            #              bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.15 * min(y_fit)),
            #              fontsize=13, arrowprops=dict(arrowstyle="-"))
            # print(popt)
            # print(pcov)
            # print(stdevs)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.legend(loc=0)
        # plt.title("Übergang von Exponentieller Dämpfung zur Linearen Dämpfung")
        # for val, err in zip(popt,stdevs):
        #    plt.annotate(rf"$g = j{round(val,deci)} \pm {round_up(err,deci)}" + r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$" , xy=(min(x_data), min(y_fit)),
        #            bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.05 * min(y_fit)),
        #            fontsize=13, arrowprops=dict(arrowstyle="-"))

        ##chisq, pval = chisquare(self.data[y]/10,objective(self.data[x],*popt))

        # plt.annotate(rf"P-Value = {round(pval,deci)}" , xy=(min(x_data), min(y_fit)),
        #         bbox={'facecolor': '#616161', 'alpha': 0.85}, xytext=(2.00 * min(x_data), 1.25 * min(y_fit)),
        #         fontsize=13, arrowprops=dict(arrowstyle="-"))
        # plt.show()
        plt.savefig(
            f"./Output/{self.name}/fit_of_{x}_{y}_mess_nr_{self.load_data_counter}.png",
            dpi=400,
        )
        # print(chisq(objective,self.data[x],self.data[y]/10,popt,self.data[f"d{y}"]/10))
        # print(chisq_stat(objective,self.data[x],self.data[y]/10,popt,self.data[rf"\sigma_{{{y}}}"]/10))

    # need to be changed to be more general currently it is tailored to g Erdbeschleunigung
    def plot_linear_reg(
        self,
        x,
        y,
        labels,
        show_lims=False,
        xerr=False,
        offset=[0, 0],
        units=[r"$s^2$", r"$s^2$"],
        title="",
    ):
        """Calculates the linear regression of data given and plot it

        :x: TODO
        :y: TODO
        :returns: TODO

        """
        plt.cla()
        plt.scatter(x, y, c="b", marker=".", label="Data")
        try:
            if xerr:
                plt.errorbar(
                    x,
                    y,
                    xerr=self.data[f"d{x.name}"],
                    yerr=self.data[f"d{y.name}"],
                    fmt="none",
                    capsize=3,
                )
            else:
                plt.errorbar(x, y, yerr=self.data[f"d{y.name}"], fmt="none", capsize=3)
        except Exception as e:
            print(e)

        x_fit = np.linspace(min(x), max(x), 100)
        lmod = LinearModel()
        pars = lmod.guess(y, x=x)
        out = lmod.fit(y, pars, x=x)
        plt.plot(x_fit, out.eval(x=x_fit), "r-", label="best fit")
        print(out.fit_report(min_correl=0.25))
        print(out.values)
        for i, (name, param) in enumerate(out.params.items()):
            print("{:7s} {:11.5f} {:11.5f}".format(name, param.value, param.stderr))
            deci = -orderOfMagnitude(param.stderr)
            print(deci)
            if deci < 0:
                deci = 1
            plt.text(
                s=rf"${name} = {format(round(param.value, deci), f'.{deci}f')} \pm {round_up(param.stderr, deci)}$ "
                + units[i],
                bbox={"facecolor": "#616161", "alpha": 0.85},
                x=(max(x) - min(x)) / 100 * (5 + offset[0]) + min(x),
                y=(max(y) - min(y)) * (offset[1] / 100 + (i + 1) / 7) + min(y),
                fontsize=10,
            )

        if show_lims:
            upper = Parameters()
            lower = Parameters()
            for name, param in out.params.items():
                upper.add(name, value=(param.value + param.stderr))
                lower.add(name, value=(param.value - param.stderr))
            plt.plot(x_fit, out.eval(params=upper, x=x_fit), label="Obere Schranke")
            plt.plot(x_fit, out.eval(params=lower, x=x_fit), label="Untere Schranke")
        # plt.annotate(
        #     rf"${name} = {format(round(param.value, deci), f'.{deci}f')} \pm {round_up(param.stderr, deci)}" + r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$",
        #     xy=(x[0], y[0]),
        #     bbox={'facecolor': '#616161', 'alpha': 0.85},
        #     xytext=((max(x) - min(x)) / 100 * (5 + offset[0]) + min(x),
        #             (max(y) - min(y)) * (offset[1] / 100 + (i + 1) / 7) + min(y)),
        #     fontsize=10, arrowprops=dict(arrowstyle="-"))
        # for i, val, err in zip(count(), popt, stdevs):
        #     plt.annotate(
        #         rf"$R_i = {format(round(val, deci), f'.{deci}f')} \pm {round_up(err, deci)}" + r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$",
        #         xy=(x[0], y[0]),
        #         bbox={'facecolor': '#616161', 'alpha': 0.85},
        #         xytext=((max(x) - min(x)) / 100 * 5 + min(x), (max(y) - min(y)) / 7 * (i + 1) + min(y)),
        #         fontsize=13, arrowprops=dict(arrowstyle="-"))

        print(out.redchi)

        # X = x.to_numpy().reshape(-1, 1)
        # Y = y.to_numpy().reshape(-1, 1)
        # reg = LinearRegression().fit(X, Y)
        # x_fit = np.linspace(min(X), max(X), 100)
        # y_fit = reg.predict(x_fit)
        # popt, pcov = curve_fit(linear_func, x, y, sigma=self.data[f"d{y.name}"], absolute_sigma=True)
        #
        # # this section is for testing where x and y have errors in them
        # data = RealData(x, y, self.data[f"d{x.name}"], self.data[f"d{y.name}"])
        # model = Model(linear_odr)
        #
        # odr = ODR(data, model, [-0.1, 1.5])
        # odr.set_job(fit_type=2)
        # output = odr.run()
        # print("Section using ODR")
        # print(output.beta)
        # print(output.sd_beta)
        # # end testing section
        # stdevs = np.sqrt(np.diag(pcov))
        # print("Section using Curve_Fit")
        # print(popt)
        # print(stdevs)
        #
        # if show_lims:
        #     plt.plot(x_fit, linear_odr(output.beta + output.sd_beta, x_fit), label="Obere Schranke")
        #     plt.plot(x_fit, linear_odr(output.beta - output.sd_beta, x_fit), label="Untere Schranke")
        # # for i, val, err in zip(count(), popt, stdevs):
        # #     plt.annotate(
        # #         rf"$R_i = {format(round(val, deci), f'.{deci}f')} \pm {round_up(err, deci)}" + r"\frac{\mathrm{m}}{\mathrm{s}^{2}}$",
        # #         xy=(x[0], y_fit[0]),
        # #         bbox={'facecolor': '#616161', 'alpha': 0.85},
        # #         xytext=((max(x) - min(x)) / 100 * 5 + min(x), (max(y_fit) - min(y_fit)) / 7 * (i + 1) + min(y_fit)),
        # #         fontsize=13, arrowprops=dict(arrowstyle="-"))
        # deci = 3
        # plt.annotate(
        #     rf"$R_i = {format(abs(round(output.beta[0], deci)), f'.{deci}f')} \pm {round_up(output.sd_beta[0], deci)}" + r" \Omega$",
        #     xy=(x[0], y_fit[0]),
        #     bbox={'facecolor': '#616161', 'alpha': 0.85},
        #     xytext=((max(x) - min(x)) / 100 * 5 + min(x), (max(y_fit) - min(y_fit)) / 7 + min(y_fit)),
        #     fontsize=13, arrowprops=dict(arrowstyle="-"))
        #
        # deci = 4
        # plt.annotate(
        #     rf"$U_0 = {format(round(output.beta[1], deci), f'.{deci}f')} \pm {round_up(output.sd_beta[1], deci)}$" + r" V",
        #     xy=(x[0], y_fit[0]),
        #     bbox={'facecolor': '#616161', 'alpha': 0.85},
        #     xytext=((max(x) - min(x)) / 100 * 5 + min(x), (max(y_fit) - min(y_fit)) / 7 * (1 + 1) + min(y_fit)),
        #     fontsize=13, arrowprops=dict(arrowstyle="-"))
        #
        # plt.annotate("$R^{2}$:" + str(floor(reg.score(X, Y) * 10000) / 100) + "%", xy=(x_fit[0], y_fit[0]),
        #              bbox={'facecolor': '#616161', 'alpha': 0.85},
        #              xytext=((max(x) - min(x)) / 100 * 35 + min(x), (max(y_fit) - min(y_fit)) / 7 * 6 + min(y_fit)),
        #              fontsize=13, arrowprops=dict(arrowstyle="-"))
        #
        # plt.plot(x_fit, y_fit, c='r', label="Linearer Fit")
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title)
        # plt.title("Klemmspannung in Abhängigkeit vom Strom Messungen einer Batterie")
        plt.legend(loc=0)
        # plt.show()
        plt.savefig(
            f"lin_reg_{x.name}_{y.name}_messreihe_{self.load_data_counter}.png", dpi=400
        )

    def chisq_stat(self, f, x_data, y_data, popt, sigma):
        prediction = f(x_data, *popt)
        r = y_data - prediction
        chisq = np.sum((r / sigma) ** 2)
        return chisq

    def find_min_sign_changes(self, df, identifier):
        vals = df[identifier].values
        abs_sign_diff = np.abs(np.diff(np.sign(vals) + (vals == 0)))
        # idx of first row where the change is
        change_idx = np.flatnonzero(abs_sign_diff == 2)
        # +1 to get idx of second rows in the sign change too
        change_idx = np.stack((change_idx, change_idx + 1), axis=1)

        # now we have the locations where sign changes occur. We just need to extract
        # the `value` values at those locations to determine which of the two possibilities
        # to choose for each sign change (whichever has `value` closer to 0)

        # min_idx = np.abs(vals[change_idx]).argmin(1)
        # return df.iloc[change_idx[range(len(change_idx)), min_idx]]
        min_idx = np.abs(vals[change_idx]).argmin(1)
        return df.iloc[change_idx[range(len(change_idx)), min_idx]]

    def probe_for_zeros(self, start, end, identifier):
        # TODO remove t filter
        pz = self.find_min_sign_changes(self.data, identifier)["t"]
        pz = pz[pz.between(start, end)]
        print(pz)
        self.data = self.data[self.data["t"].between(start, end)]
        return pz

    def plot_function(self):
        pass


def linear_func(x, g):
    """Plain old linear function

    :x: TODO
    :returns: TODO

    """
    return 4 * np.pi * np.pi * x / g

    # print("Integrals")
    # I = trapz(P.data['p'], P.data['V']) / passes * hpacm3tosi
    # print(I)
    # c = 1
    # for i in np.linspace(244, 2432, 10):
    #     print(i)
    #     df = P.data[(P.data.t < i)]
    #     I2 = trapz(df['p'], df['V']) / c * hpacm3tosi
    #     c += 1
    #     print(I2)
    # print("End Integrals")
    # drehzahl = 1 / (2432 / 10000)
    # print(f"Drehzahl aus Daten {drehzahl}")
    # print(f"Mechanische Leistung {drehzahl * 1.87}")
    # print(f"Heizleistung Leistung 107.5 W")
    # print(f"Wirkungsgrad {drehzahl * 1.87 / 107.5}")
    #
    # P.data = P.data[(2 < P.data.f) & (P.data.f < 7)]
    # P.plot_data("f", "Vf", labels=["f / s$-1$", "V(f) / cm$^3$"])


#
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
