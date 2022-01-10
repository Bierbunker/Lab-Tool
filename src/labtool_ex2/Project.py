# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

# import scipy as sp
# import sympy
import pandas
import math
import re
import sympy as simp
from collections.abc import Iterable
from sympy.utilities.iterables import ordered
import matplotlib
from typing import Callable, Union, Any
from scipy import integrate
from scipy.signal import find_peaks

matplotlib.use("Agg")
from matplotlib.legend import _get_legend_handles_labels
import matplotlib.pyplot as plt
from lmfit import Parameters
from lmfit import conf_interval2d

# from math import floor
# from math import sqrt
from sympy.interactive import printing
from lmfit.models import ExpressionModel
from lmfit.models import LinearModel
from uncertainties import ufloat
from numpy.typing import ArrayLike

from pathlib import Path


from sympy import Matrix, hessian, lambdify
from sympy import latex

from .Equation import Equation  # relative import

DataFrameLike = UArrayLike = ItDepends = Any

printing.init_printing(use_latex=True)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


class Project:
    def __init__(self, name, global_variables=dict(), global_mapping=dict(), font=10):
        self.name = name
        self.equations = dict()
        self.gm = global_mapping
        self.gv = global_variables
        self.figure = plt.figure()
        self.data_path = list()
        # BEGIN these are currently not used but maybe in the future
        self.messreihen_dfs = list()
        self.working_dfs = list()
        self.local_ledger = dict()
        # END these are currently not used but maybe in the future
        self.raw_data = pandas.DataFrame()
        self.data = pandas.DataFrame()
        self.dfs = dict()
        self.load_data_counter = 0
        rcsettings = {
            "font.size": font,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{lmodern} \usepackage{siunitx}",
            "font.family": "Latin Modern Roman",
        }
        plt.rcParams.update(rcsettings)

        p = Path(f"./Output/{name}/")
        p.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"""This is the {self.name} Project \n
                following mappings happen in this project {self.gm}"""

    def load_data(self, path, loadnew=False, clean=True):
        """
        path is the file containing csv data
        loadnew resets the currently loaded data dataframe should be used when importing many files
        clean should not be touched if you don't know what you are doing contact max if raw data is need"""
        print("\n\nLoading Data from: " + path)
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

    def create_Eq(self, *args, **kwargs):
        """This is the recommended way to create Equations in a Project,
        but feel free to instantiate Equation directly
        """
        # print(args)
        # print(kwargs)
        mappings = self.gm
        if "mapping" in kwargs:
            mappings.update(kwargs["mapping"])
            kwargs["mapping"] = mappings
        else:
            kwargs["mapping"] = self.gm
        if "figure" not in kwargs or not kwargs["figure"]:
            kwargs["figure"] = self.figure
        if "dataframe" not in kwargs or kwargs["dataframe"].empty:
            kwargs["dataframe"] = self.data
        kwargs["project_name"] = self.name
        # print(args)
        # print(kwargs)
        eq = Equation(*args, **kwargs)
        self.equations[eq.var_name] = eq
        return eq

    def find_possible_zero(self, identifier):
        return self.data[~self.data[identifier].astype(bool)]

    def interpolate(self):
        # TODO hard take min max of each free_symbols and make smooth interpolation of each variable
        pass

    def normalize(self):
        """converts none si units into si units
        :returns: TODO

        """
        pass

    # BEGIN should be moved to Equation
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

    # END should be moved to Equation

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

    def print_table(self, df, name=None):
        """Tries to export dataframe to latex table

        :df: TODO
        :returns: TODO

        """
        print(df.columns)
        colnames = list()
        for colname in df.columns:
            if colname[0] is "d" and colname[1:] in self.gm:
                unit = " / " + self.gv[colname[1:]]
                colname = self.gm[colname[1:]]
                colnames.append(r"$\Delta " + colname + "$" + unit)
                continue
            if colname in self.gm:
                unit = " / " + self.gv[colname]
                colname = self.gm[colname]
                colnames.append("$" + colname + "$" + unit)
                continue
            if "d" in colname and "\\" not in colname:
                colname = colname.replace("d", "\\Delta ")
            if len(colname) > 1 and "\\" not in colname:
                colname = "\\" + colname

            colname = "$" + colname + "$"
            colnames.append(colname)

        df.columns = colnames
        if name:
            with open(f"./Output/{self.name}/messreihe_{name}.tex", "w") as tf:
                tf.write(df.to_latex(escape=False))
        else:
            with open(f"./Output/{self.name}/messreihe_{df.name}.tex", "w") as tf:
                tf.write(df.to_latex(escape=False))

    def write_table(
        self,
        content: DataFrameLike,
        path: str,
        environ: str = "tblr",
        colspec: Union[str, list[str]] = "",
        inner_settings: list[str] = [],
        columns: Union[bool, list[str]] = False,
        index: bool = False,
        format_spec: Union[None, str] = None,
        uarray: bool = False,
        sisetup: list[str] = [],
        hlines_old: bool = False,
        msg: bool = False,
    ) -> str:
        """
        Copied from Andreas Zach ;) aber an die Project Klasse angepasst
        Create a tex-file with a correctly formatted table for LaTeX-package 'tabularray' from the given
        input content. Return the created string.
        Mandatory parameters:
        -> content\t\t\tmust be convertible to pandas.DataFrame
        -> path\t\t\tname (or relative path) to tex-file for writing the table to, can be left empty
        \t\t\t\tin order to just return the string
        Optional parameters:
        -> environ='tblr'\t\ttblr environment specified in file 'tabularray-environments.tex',
        \t\t\t\toptional 'tabular' to use a standard LaTeX-table
        -> colspec=''\t\tcolumn specifier known from standard LaTeX tables (only suited for tabularray!),
        \t\t\t\tone string or list of strings
        -> inner_settings=[]\tadditional settings for the tabularray environment (see documentation), input as list of strings
        \t\t\t\tif standard 'tabular' environment is used, the column specifiers can be put here as one entry of the list
        -> columns=False\t\twrite table header (first row), input as list of strings or boolean
        -> index=False\t\tboolean if indices of rows (first column) should be written to table
        -> format_spec=None\t\tfloat formatter (e.g. .3f) or None if floats should not be formatted specifically
        -> uarray=False\t\tboolean if input was created with uncertainties.unumpy.uarray
        -> sisetup=[]\t\tlist with options for \\sisetup before tblr gets typeset
        -> hlines_old=False\t\tif standard tabular environment is used, this can be set to True to draw all hlines
        -> msg=False\t\tboolean if the reformatted DataFrame and the created string should be printed to the console
        """
        # input must be convertible to pandas.DataFrame
        df = pandas.DataFrame(content)

        # format_specifier
        formatter = f"{{:{format_spec}}}".format if format_spec is not None else None

        # append column specifier to inner settings
        if colspec:
            if isinstance(colspec, list):
                colspec = "".join(colspec)

            inner_settings.append(f"colspec={{{colspec}}}")
            # double curly braces produce literal curly brace in f string
            # three braces: evaluation surrounded by single braces

        # prepare columns for siunitx S columns
        # columns could be bool or Iterable[str]

        # check if columns has any truthy value
        if columns:

            # identity check with 'is' because columns could be a non-empty container
            # alternative: isinstance(columns, bool)
            if columns is True:
                columns = df.columns.tolist()

            # non-empty container
            else:
                # check if right amount of column labels was provided
                if len(columns) != len(df.columns):
                    raise IndexError(
                        "'content' had a different amount of columns than provided 'columns'"
                    )
                else:
                    # update columns of DataFrame
                    df.columns = columns  # type: ignore

            # if columns was True, it's now a list
            # else it's still the provided Iterable with correct length
            # make strings safe for tabularry's siunitx S columns
            colnames = list()
            for colname in columns:
                if colname in self.gm:
                    colname = self.gm[colname]
                    colnames.append(colname)
                    continue
                if "d" in colname and "\\" not in colname:
                    colname = colname.replace("d", "\\Delta ")
                if len(colname) > 1 and "\\" not in colname:
                    colname = "\\" + colname

                colname = "$" + colname + "$"
                colnames.append(colname)
            columns = colnames
            columns = [f"{{{{{{{col}}}}}}}" for col in columns]  # type: ignore

        # if falsy value, it should be False altogether
        else:
            columns = False

        # strings
        sisetup_str = ", ".join(sisetup)
        inner_settings_str = ",\n".join(inner_settings)
        hlines_str = "\\hline" if hlines_old else ""
        df_str: str = df.to_csv(
            sep="&",
            line_terminator=f"\\\\{hlines_str}\n",  # to_csv without path returns string
            float_format=formatter,
            header=columns,
            index=index,
        )  # type: ignore

        if uarray:
            # delete string quotes
            df_str = df_str.replace('"', "")

            # replace +/- with +-
            df_str = re.sub(r"(\d)\+/-(\d)", r"\1 +- \2", df_str)

            # delete parantheses and make extra spaces if exponents
            df_str = re.sub(r"\((\d+\.?\d*) \+- (\d+\.?\d*)\)e", r"\1 +- \2 e", df_str)

        # create complete string
        complete_str = f"\\sisetup{{{sisetup_str}}}\n\n" if sisetup_str else ""
        complete_str += (
            f"\\begin{{{environ}}}{{{inner_settings_str}}}{hlines_str}\n"
            f"{df_str}"
            f"\\end{{{environ}}}"
        )

        # write to file if path provided
        if path:
            # open() does not encode in utf-8 by default
            with open(path, "w", encoding="utf-8") as f:
                f.write(complete_str)

        # message printing
        if msg:
            # pd.options
            options.display.float_format = formatter

            print(
                f"Wrote pandas.DataFrame\n\n{df}\n\n"
                f"as tabularray environment '{environ}' to file '{path}'\n\n\n"
                f"output:\n\n{complete_str}"
            )

        return complete_str

    def figure_legend(self, **kwargs):
        self.figure.legend(*_get_legend_handles_labels(self.figure.axes), **kwargs)

    def ax_legend_all(self, **kwargs):
        self.figure.get_axes()[0].legend(
            *_get_legend_handles_labels(self.figure.axes), **kwargs
        )

    def fig_legend(self, **kwdargs):

        # generate a sequence of tuples, each contains
        #  - a list of handles (lohand) and
        #  - a list of labels (lolbl)
        tuples_lohand_lolbl = (
            ax.get_legend_handles_labels() for ax in self.figure.axes
        )
        # e.g. a figure with two axes, ax0 with two curves, ax1 with one curve
        # yields:   ([ax0h0, ax0h1], [ax0l0, ax0l1]) and ([ax1h0], [ax1l0])

        # legend needs a list of handles and a list of labels,
        # so our first step is to transpose our data,
        # generating two tuples of lists of homogeneous stuff(tolohs), i.e
        # we yield ([ax0h0, ax0h1], [ax1h0]) and ([ax0l0, ax0l1], [ax1l0])
        tolohs = zip(*tuples_lohand_lolbl)

        # finally we need to concatenate the individual lists in the two
        # lists of lists: [ax0h0, ax0h1, ax1h0] and [ax0l0, ax0l1, ax1l0]
        # a possible solution is to sum the sublists - we use unpacking
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)

        # call fig.legend with the keyword arguments, return the legend object

        return self.figure.legend(handles, labels, **kwdargs)

    def find_axes(self, xlabel=None, ylabel=None):
        if xlabel and ylabel:
            for ax in self.figure.get_axes():
                print(f"testing {ax.get_xlabel()} and {ax.get_ylabel()}")
                if (ax.get_xlabel() == xlabel or ax.get_xlabel() == "") and (
                    ax.get_ylabel() == ylabel or ax.get_ylabel() == ""
                ):
                    return ax
            return None
        elif xlabel:
            for ax in self.figure.get_axes():
                if ax.get_xlabel() == xlabel:
                    return ax
            return None
        elif ylabel:
            for ax in self.figure.get_axes():
                if ax.get_ylabel() == ylabel:
                    return ax
            return None

    def set_x_y_label(self, ax, x, y, color="k"):
        unitx = self.gv[x]
        unity = self.gv[y]
        if x in self.gm:
            x = self.gm[x]
        if y in self.gm:
            y = self.gm[y]
        xlabel = rf"${x}$ / {unitx}"
        ylabel = rf"${y}$ / {unity}"
        if not ax.get_ylabel():
            ax.set_ylabel(ylabel, color=color)
        if not ax.get_xlabel():
            ax.set_xlabel(xlabel, color=color)

        # tries to find a compatible axes, axes are compatible if they have the same x and y labels
        axtra = self.find_axes(xlabel=xlabel, ylabel=ylabel)
        # print(f"current axes x = {xlabel} & y = {ylabel}")
        if axtra:
            # print(f"axes found for x = {xlabel} & y = {ylabel}")
            return axtra

        if ax.get_xlabel() != xlabel and ax.get_ylabel() != ylabel:
            self.figure.subplots_adjust(bottom=0.2, right=0.85)
            axtra = self.figure.add_axes(ax.get_position())
            axtra.patch.set_visible(False)

            axtra.yaxis.set_label_position("right")
            axtra.yaxis.set_ticks_position("right")

            axtra.spines["bottom"].set_position(("outward", 35))
            axtra.set_xlabel(xlabel, color=color)
            axtra.set_ylabel(ylabel, color=color)
            return axtra
        elif ax.get_xlabel() != xlabel:
            axtra = ax.twiny()
            axtra.set_xlabel(xlabel, color=color)
            return axtra
        elif ax.get_ylabel() != ylabel:
            axtra = ax.twinx()
            axtra.set_ylabel(ylabel, color=color)
            return axtra

        return ax

    def plot_data(
        self,
        axes,
        x,
        y,
        label,
        style="r",
        errors=False,
    ):
        raw_x = self.data[x]
        raw_y = self.data[y]
        axes = self.set_x_y_label(ax=axes, x=x, y=y, color=style)

        if errors:
            errs = dict()
            try:
                name = "d" + x
                errs["xerr"] = self.data[name].values
            except Exception as e:
                print(f"No xerr for {x} found")
            try:
                name = "d" + y
                errs["yerr"] = self.data[name].values
            except Exception as e:
                print(f"No yerr for {y} found")
            axes.errorbar(raw_x, raw_y, fmt="none", capsize=3, **errs)

        axes.scatter(
            raw_x,
            raw_y,
            c=style,
            marker=".",
            s=39.0,
            label=label,
        )

        # axes.legend(loc=0)

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

    def expr_to_np(self, expr):
        return simp.lambdify(tuple(expr.free_symbols), expr, "numpy")

    def apply_df(self, expr):
        function_data = dict()
        if hasattr(expr, "lhs"):
            for var in expr.free_symbols:
                var = str(var)
                if var is not expr.lhs:
                    series = self.data[var]
                    function_data[var] = series.to_numpy()
            return self.expr_to_np(expr)(**function_data)
        else:
            for var in expr.free_symbols:
                var = str(var)
                series = self.data[var]
                function_data[var] = series.to_numpy()
            return self.expr_to_np(expr)(**function_data)

    def apply_df_err(self, expr):
        function_data = dict()
        err_expr = self.error_func(expr=expr)
        for var in err_expr.free_symbols:
            var = str(var)
            series = self.data[var]
            function_data[var] = series.to_numpy()
        return self.expr_to_np(expr=err_expr)(**function_data)

    def error_func(self, expr):
        vs = list(ordered(expr.free_symbols))

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0
        errs = " ".join([f"d{s}" for s in vs])
        er = simp.symbols(errs)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(expr, vs), er):
            e_func = e_func + abs(c) * s
        return e_func

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

    def rms_wave(self, x, y, f):
        I1 = integrate.simpson(y ** 2, x)
        rms = np.sqrt(I1 / (max(x) - min(x)))
        return rms

    def avr_wave(self, x, y, f):
        I1 = integrate.trapz(abs(y), x)
        avr = I1 / (max(x) - min(x))
        return avr

    def find_phase_difference(self, t, v, w, f):
        pks = find_peaks(x=v, prominence=1)
        print(pks)
        pks = find_peaks(x=w, prominence=1)
        print(pks)
        mag1, phase1 = self.find_phase_and_gain(t, v, f)
        mag2, phase2 = self.find_phase_and_gain(t, w, f)
        return phase1 - phase2

    def find_phase_and_gain(self, t, v, f):
        a_sum = 0  # now do it all again for the output wave
        b_sum = 0
        w = 2 * np.pi * f
        for time, voltage in zip(t, v):
            a_sum = a_sum + voltage * np.cos(w * time)
            b_sum = b_sum + voltage * np.sin(w * time)
        a = a_sum * 2 / len(t)
        b = b_sum * 2 / len(t)
        mag = np.sqrt(a ** 2 + b ** 2)
        phase = np.arctan2(a, b)
        print(mag)
        print(phase)
        return (mag, phase)

    def plot_function(self, axes, x, y, expr, label, errors=False, style="r"):
        # fig, ax = self.get_fig_ax(figure, toggle_add_subplot)
        x_data = self.data[x]
        # x_continues_plot_data = np.linspace(x_data.min(), x_data.max(), 1000)
        # function_data = dict()
        # print(str(expr.free_symbols))
        # for var in expr.free_symbols:
        #     var = str(var)
        #     if var is not x and var is not expr.lhs:
        #         series = self.data[var]
        #         function_data[var] = series.to_numpy()
        # function_data[x] = x_continues_plot_data
        y_data = self.apply_df(expr)
        # p22nt(y_data)

        # try:
        self.set_x_y_label(axes, x, y)
        # except AttributeError as e:
        #     self.set_x_y_label(axes, x, str(expr))

        axes.plot(
            x_data,
            y_data,
            linestyle="-",
            color=style,
            label=label,
        )
        if errors:
            if f"d{y}" in self.data:
                dely = self.data[f"d{y}"]
                axes.fill_between(
                    x_data, y_data - dely, y_data + dely, color=style, alpha=0.4
                )
        # axes.legend(loc=0)

    def plot_fit(
        self,
        axes,
        x,
        y,
        eqn,
        style="r",
        label="fit",
        use_all_known=False,
        offset=[0, 0],
        guess=None,
        bounds=None,
        add_fit_params=True,
        granularity=10000,
        gof=False,
        sigmas=1,
    ):
        """axes current main axes to plot on
        x string of independent variable
        y string of dependent variable
        eqn expression or equation sympy object which results in y
        style string what should I say see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        label of the fit in the legend
        use_all_known flag tries to use all findable data for the unkowns for clarification call max
        offset relative offset position in the figure for the fit info
        guess Mandatory everybody got to guess
        add_fit_params flag wether or not the fitted parameters should be printed in the figure
        granularity is the granularity of the continues x data for the fit curve
        gof flag if goodness of fit should be printed
        """
        x_data = self.data[x]
        y_data = self.data[y]
        print(eqn)
        x_continues_plot_data = np.linspace(x_data.min(), x_data.max(), granularity)
        print(x_continues_plot_data)
        y_fit = y_data.values.reshape(1, -1)
        independent_vars = dict()
        cont_independent_vars = dict()
        if hasattr(eqn, "lhs"):
            eqn = eqn[1]
        if use_all_known:
            vars = [str(sym) for sym in eqn.free_symbols if str(sym) != y]
            mod = ExpressionModel(
                str(eqn),
                independent_vars=vars,
            )
            for var_ in vars:
                independent_vars[var_] = self.data[var_].values.reshape(1, -1)
                cont_independent_vars[var_] = np.linspace(
                    self.data[var_].min(), self.data[var_].max(), granularity
                )
        else:
            mod = ExpressionModel(
                str(eqn),
                independent_vars=[x],
            )
            x_fit = x_data.values.reshape(1, -1)
            independent_vars[x] = x_fit
            cont_independent_vars[x] = np.linspace(
                x_data.min(), x_data.max(), granularity
            )

        if guess:
            pars = mod.make_params(**guess)
        else:
            # pars = mod.guess(raw_y, x=raw_x)
            # TODO custom error needs to be implemented
            print("No guess was passed")
            raise Exception
        if bounds:
            for bound in bounds:
                mod.set_param_hint(
                    name=bound["name"], min=bound["min"], max=bound["max"]
                )

        # print(independent_vars)
        # print(pars)

        if f"d{y}" in self.data:
            weights = 1 / (self.data[f"d{y}"])
            weights = weights.values.reshape(1, -1)
            out = mod.fit(
                y_fit, pars, weights=weights, scale_covar=False, **independent_vars
            )
        else:
            # out = mod.fit(y_fit, pars,scale_covar=False, **independent_vars)
            out = mod.fit(y_fit, pars, **independent_vars)
        axes = self.set_x_y_label(ax=axes, x=x, y=y)
        if add_fit_params:
            paramstr = "\n".join(
                [
                    rf"${self.gm[name]} = {ufloat(param.value,sigmas*param.stderr)}$ "
                    + self.gv[name]
                    for (name, param) in out.params.items()
                ]
            )
            if gof:
                paramstr += "\n" + rf"$\chi^2 = {out.chisqr}$ "
                paramstr += "\n" + rf"$\chi^2_{{\nu}} = {out.redchi}$ "
            #            for i, (name, param) in enumerate(out.params.items()):
            #                uparam = ufloat(param.value, param.stderr)
            #                paramstr += rf"${self.gm[name]} = {uparam}$ " + self.gv[name] + "\n"
            #   axes.text(
            #       s=rf"${self.gm[name]} = {uparam}$ " + self.gv[name],
            #       bbox={"facecolor": style, "alpha": 0.75},
            #       # bbox={"facecolor": "#616161", "alpha": 0.85},
            #       x=(max(x_data) - min(x_data)) / 100 * (5 + offset[0]) + min(x_data),
            #       y=(max(y_data) - min(y_data)) * (offset[1] / 100 + (i + 1) / 13)
            #       + min(y_data),
            #       fontsize=10,
            #   )

            axes.text(
                s=paramstr,
                bbox={"facecolor": style, "alpha": 0.60},
                # bbox={"facecolor": "#616161", "alpha": 0.85},
                x=(max(x_data) - min(x_data)) / 100 * (5 + offset[0]) + min(x_data),
                y=(max(y_data) - min(y_data)) * (offset[1] / 100) + min(y_data),
                fontsize=10,
            )
        resp = np.array(out.eval(**cont_independent_vars))
        print(resp)
        # prevl = 0.5790
        # prevn = 1.78560969693395
        # prevd = 1
        # for l,n in zip(x_continues_plot_data,resp):
        #     if 0.5790 <l < 0.5792:
        #         d = (n-prevn)/(l-prevl)
        #         print(l,n, l-prevl,n-prevn,(n-prevn)/(l-prevl), d - prevd)
        #         prevl= l
        #         prevn=n
        #         prevd = d
        # out = model.fit(data, params, x=x)
        # cx, cy, grid = conf_interval2d(out, out, "a1", "t2", 30, 30)
        # ctp = axes[0].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        # fig.colorbar(ctp, ax=axes[0])
        dely = sigmas * out.eval_uncertainty(**cont_independent_vars)
        # axes.plot(x, data)
        # axes.plot(x, out.best_fit)
        axes.fill_between(
            x_continues_plot_data, resp - dely, resp + dely, color=style, alpha=0.4
        )
        axes.plot(x_continues_plot_data, resp, style, label=f"{label} fit")
        # axes.legend(loc=0)
        print(out.fit_report(min_correl=0.25))

    def savefig(self, name, clear=True):
        self.figure.savefig(f"./Output/{self.name}/" + name, dpi=400)
        if clear:
            self.figure.clear()
            ax = self.figure.add_subplot()
            return ax

    def add_text(self, axes, keyvalue, offset=[0, 0]):
        # maxes = max([_.zorder for _ in axes.get_children()])
        # print(maxes)
        maxes = axes
        for i, (name, param) in enumerate(keyvalue.items()):
            uparam = param
            xmin, xmax = maxes.get_xlim()
            ymin, ymax = maxes.get_ylim()
            maxes.text(
                s=rf"${self.gm[name]} = {uparam}$ " + self.gv[name],
                bbox={"facecolor": "#616161", "alpha": 0.85},
                x=(xmax - xmin) / 100 * (5 + offset[0]) + xmin,
                y=(ymax - ymin) * (offset[1] / 100 + (i + 1) / 7) + ymin,
                fontsize=10,
            )
