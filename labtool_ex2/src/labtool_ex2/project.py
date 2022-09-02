# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

# import scipy as sp
# import sympy
import pandas
import re
import io
import sys
import sympy as simp
import inspect
from collections.abc import Iterable
from pandas import DataFrame
from sympy.utilities.iterables import ordered
import matplotlib
from typing import Callable, Union, Any, Sequence, Optional
from pandas._typing import (
    FilePathOrBuffer,
)
from sympy.core.expr import Expr
from scipy import integrate
from scipy.signal import find_peaks

# from lmfit import Parameters
# from lmfit import conf_interval2d
# from lmfit.models import LinearModel

from lmfit.models import ExpressionModel
from uncertainties import ufloat

# from uncertainties.core import format_num
from .helpers import round_up

from pathlib import Path
from .LatexPrinter import latex
import labtool_ex2.symbol as s

# from .symbol import _custom_var

from sympy import Matrix, hessian, lambdify, symbols
from sympy.interactive import printing
from matplotlib.legend import Legend

# dont fucking move this line

matplotlib.use("Agg")
from matplotlib.legend import _get_legend_handles_labels
import matplotlib.pyplot as plt


DataFrameLike = UArrayLike = ItDepends = Any

printing.init_printing(use_latex=True)


class Project:
    def __init__(
        self,
        name: Optional[str],
        global_variables: dict[str, str] = dict(),
        global_mapping: dict[str, str] = dict(),
        font: int = 10,
        infer: bool = True,
    ):
        self.name: Optional[str] = name
        # self.equations = dict()
        self.gm: dict[str, str] = global_mapping
        self.gv: dict[str, str] = global_variables
        self._infer: bool = infer
        self.figure: plt.Figure = plt.figure()
        # self.data_path = list() glob internal PATH variable
        # BEGIN these are currently not used but maybe in the future
        self.messreihen_dfs: list = list()
        self.working_dfs: list = list()
        self.local_ledger: dict = dict()
        # END these are currently not used but maybe in the future
        self.raw_data: DataFrame = DataFrame()
        self.data: DataFrame = DataFrame()
        self.dfs: dict[str, DataFrame] = dict()
        self.load_data_counter: int = 0
        if name is not None:
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

        # inject variables into global namespace can be improved if necessary however sympy does the same thing
        s._custom_var(list(global_variables), project=self)
        # simp.var(list(global_variables), cls=Symbol, df=self.data)

    # import sys
    # import ctypes

    def vload(self, names: Optional[list[str]], **args):
        if names:
            s._custom_var(names=names, project=self, **args)
        else:
            s._custom_var(names=list(self.gv), project=self, **args)

    # def hack():
    #     # Get the frame object of the caller
    #     frame = sys._getframe(1)
    #     frame.f_locals['x'] = "hack!"
    #     # Force an update of locals array from locals dict
    #     ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame),
    #                                           ctypes.c_int(0))

    # def func():
    #     x = 1
    #     hack()
    #     print(x)

    # func()

    def __str__(self) -> str:
        return f"""This is the [self.name] Project \n
                following mappings happen in this project {self.gm}"""

    def load_data(
        self,
        path: Union[str, FilePathOrBuffer],
        loadnew: bool = False,
        clean: bool = True,
    ) -> None:
        """
        path is the file containing csv data
        loadnew resets the currently loaded data dataframe should be used when importing many files
        clean should not be touched if you don't know what you are doing contact max if raw data is need"""
        print("\n\nLoading Data from: " + str(path))
        if loadnew:
            self.data.drop(self.data.index, inplace=True)
            # self.data = pandas.DataFrame(data=None)
        df = pandas.read_csv(
            path, header=[0, 1], skipinitialspace=True  # type: ignore
        )  # type: DataFrame
        df.columns = pandas.MultiIndex.from_tuples(
            df.columns, names=["type", "variable"]
        )
        self.load_data_counter += 1
        self.raw_data = df.astype(float)
        if clean:
            name = Path(path).stem  # type: ignore
            self._clean_dataset(name=name)

    def _clean_dataset(self, name: str, use_min: bool = False):
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

    def find_possible_zero(self, identifier) -> DataFrameLike:
        """Finds possible zeros in the dataset on a column <identifier>"""
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
    # Currently not used for future
    def get_vardata(self, raw_data=False):
        """Simply returns the data needed to calculate the equation
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():  # type: ignore
            reg = rf"^{c}$"
            if raw_data:
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    # Currently not used for future
    def get_errdata(self, raw_data=False):
        """Simply returns the error of the variables
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():  # type: ignore
            reg = rf"^d{c}$"
            if raw_data:
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    # Currently not used for future
    def get_vardata_and_errdata(self, raw_data=False):
        """Simply returns the variables with their errors
        :returns: TODO

        """
        df = pandas.DataFrame(data=None)
        for c in self.variables.split():  # type: ignore
            reg = rf"^(d)?{c}$"
            if raw_data:
                filt = self.raw_data.droplevel("type", axis=1).filter(regex=reg)
            else:
                filt = self.data.filter(regex=reg)
            df = pandas.concat([df, filt], axis=1)
        return df

    # Currently not used for future
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

    # Currently not used for future
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

    def print_expr(self, expr: Expr):
        """Prints the expression in latex format and substitutes the variables according to the mapping"""
        print(latex(expr=expr, symbol_names=self.gm))

    # def print_table(self, expr: Expr):
    #     pass

    def print_table(self, df: DataFrameLike, name: Optional[str] = None):
        """Tries to export dataframe to latex table

        :df: TODO
        :returns: TODO

        """
        colnames = list()
        for colname in df.columns:
            if colname[0] == "d" and colname[1:] in self.gm:
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

    def print_ftable(
        self,
        df: DataFrameLike,
        name: Optional[str] = None,
        format: str = "standard",
        split: bool = False,
    ):
        df = df.u.com
        numColumns = df.shape[1]
        numRows = df.shape[0]
        output = io.StringIO()

        colnames = list()
        for colname in df.columns:
            if colname[0] == "d" and colname[1:] in self.gm:
                unit = " / " + self.gv[colname[1:]]
                colname = self.gm[colname[1:]]
                colnames.append(r"{{{$\Delta " + colname + "$" + unit + "}}}")
                continue
            if colname in self.gm:
                unit = " / " + self.gv[colname]
                colname = self.gm[colname]
                colnames.append("{{{$" + colname + "$" + unit + "}}}")
                continue
            if "d" in colname and "\\" not in colname:
                colname = colname.replace("d", "\\Delta ")
            if len(colname) > 1 and "\\" not in colname:
                colname = "\\" + colname

            colnames.append("{{{$" + colname + "$}}}")
            if split:
                colnames.append("{{{$\\Delta " + colname + "$}}}")

        esses = "S" * numColumns
        output.write("\\begin{tblr}{" + esses + "}\n")
        if split:

            def format_value(x):  # type: ignore
                fmt_x = x.__format__("")
                if "e" in fmt_x:
                    val, exp = fmt_x.split("e", 1)
                    val = (
                        val.replace("+/-", "e" + exp + " & ")
                        .replace("(", "")
                        .replace(")", "")
                        + "e"
                        + exp
                    )
                else:
                    val = fmt_x.replace("+/-", " & ")
                return val

            # colFormat = "%s|%s" % (alignment, alignment * numColumns)
            # Write header
            # output.write("\\begin{tabular}{%s}\n" % colFormat)
            output.write(" & ".join(colnames) + "\\\\\n")

            for i in range(numRows):
                output.write(
                    " & ".join([format_value(val) for val in df.iloc[i]]) + "\\\\\n"
                )
        else:
            output.write(" & ".join(colnames) + "\\\\\n")

            def format_value(x):
                fmt_x = x.__format__("S")
                s = re.sub(r"\((.*?)\)", lambda g: re.sub(r"\.", "", g[0]), fmt_x)
                return s

            for i in range(numRows):
                output.write(
                    " & ".join([format_value(val) for val in df.iloc[i]]) + "\\\\\n"
                )

        output.write("\\end{tblr}\n")
        # output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
        # Write data lines

        # Write footer
        # output.write("\\end{tabular}")
        if name:
            with open(f"./Output/{self.name}/messreihe_{name}.tex", "w") as tf:
                tf.write(output.getvalue())
        else:
            with open(f"./Output/{self.name}/messreihe_{df.name}.tex", "w") as tf:
                tf.write(output.getvalue())
        # return output.getvalue()

    def figure_legend(self, **kwargs) -> None:
        """
        This function is used to add a legend to a figure for printing.
        This is the preferred function to use.
        For kwargs see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        """
        self.figure.legend(*_get_legend_handles_labels(self.figure.axes), **kwargs)

    def ax_legend_all(self, **kwargs) -> None:
        """
        This function adds all legend to the first axes don't use this, use figure_legend
        """
        self.figure.get_axes()[0].legend(  # type:ignore
            *_get_legend_handles_labels(self.figure.axes), **kwargs
        )

    # this is only an explanation of the two functions above
    def _fig_legend(self, **kwdargs) -> Legend:

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

    def _find_axes(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
        if xlabel and ylabel:
            for ax in self.figure.get_axes():  # type:ignore
                print(f"testing {ax.get_xlabel()} and {ax.get_ylabel()}")
                if (ax.get_xlabel() == xlabel or ax.get_xlabel() == "") and (
                    ax.get_ylabel() == ylabel or ax.get_ylabel() == ""
                ):
                    return ax
            return None
        elif xlabel:
            for ax in self.figure.get_axes():  # type:ignore
                if ax.get_xlabel() == xlabel:
                    return ax
            return None
        elif ylabel:
            for ax in self.figure.get_axes():  # type:ignore
                if ax.get_ylabel() == ylabel:
                    return ax
            return None

    def _set_x_y_label(
        self, ax: plt.Axes, x: str, y: str, color: str = "k"
    ) -> plt.Axes:
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
        axtra = self._find_axes(xlabel=xlabel, ylabel=ylabel)
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
        axes: plt.Axes,
        x: str,
        y: str,
        label: str,
        style: str = "r",
        errors: bool = False,
    ) -> None:
        raw_x = self.data[x]
        raw_y = self.data[y]
        axes = self._set_x_y_label(ax=axes, x=x, y=y, color=style)

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

    def _expr_to_np(self, expr: Expr) -> Callable:
        """Converts a Sympy Expression to a numpy calculable function"""
        print(tuple(expr.free_symbols))
        return simp.lambdify(tuple(expr.free_symbols), expr, "numpy")

    def zz(self, expr: Expr):
        notfound = list()
        function_data = dict()
        # print(expr.free_symbols)
        # print(self.data.columns)
        for var in expr.free_symbols:
            if isinstance(var, s.Symbol):
                if var.name not in self.data.columns:
                    notfound.append(var.name)
                    continue

                function_data[var.name] = var.data.to_numpy()
            else:
                print("ohno")

        if notfound:
            raise Exception(f"Data for the following variables are missing: {notfound}")

        func_name = sys._getframe().f_code.co_name
        line_of_code = inspect.stack()[1][4][0]  # type: ignore
        reg = func_name + r"\((?P<params>.+)\)"
        match = re.search(reg, line_of_code)
        params = match.group("params").split(",")  # type: ignore
        new_var = params[0]
        self.data[new_var] = self._expr_to_np(expr=expr)(**function_data)
        s._custom_var(list(new_var), project=self)
        # TODO use physipy for unit translation and calculation so that units are automatically calculated

        # do = [var.name in self.data.columns for var in expr.free_symbols]
        # names = [var.name for var in expr.free_symbols]
        # if all(name in self.data.columns for name in names):
        #     print(names)

    def apply_df(
        self,
        expr: Expr,
        infer: bool | None = None,
        name: str | None = None,
        errors: bool = False,
    ) -> DataFrameLike:
        """Uses expr and internal data to calculate the value described by expr"""

        # def varnameis(v):
        #     d = globals()
        #     return [k for k in d if d[k] == v]

        # def varname(v, scope=None):
        #     d = globals() if not scope else vars(scope)
        #     return [k for k in d if d[k] == v]

        if infer is None:
            infer = self._infer

        params = list()
        if infer:
            # nice hack which finds the name of the parameters which called this function
            func_name = sys._getframe().f_code.co_name
            line_of_code = inspect.stack()[1][4][0]  # type: ignore
            reg = func_name + r"\((?P<params>.+)\)"
            match = re.search(reg, line_of_code)
            params = match.group("params").split(",")  # type: ignore

        if name is not None:
            params[0] = name
        # print(newvar.split("(")[1].split(")")[0])
        # frame = inspect.stack()[1][0]
        # while name not in frame.f_locals:
        # frame = frame.f_back
        # if frame is None:
        # return None
        # return frame.f_locals[name]
        # print(varnameis(expr))
        function_data = dict()
        if hasattr(expr, "lhs"):
            for var in expr.free_symbols:
                var = str(var)
                if var is not expr.lhs:  # type: ignore
                    series = self.data[var]
                    function_data[var] = series.to_numpy()
        else:
            for var in expr.free_symbols:
                var = str(var)
                series = self.data[var]
                function_data[var] = series.to_numpy()

        if params:
            self.data[params[0]] = self._expr_to_np(expr)(**function_data)
            # TODO Add or reset definition of variable using f_locals no like sympy does
            simp.var(params[0])
            return self.data[params[0]]
        else:
            return self._expr_to_np(expr)(**function_data)

    def apply_df_err(self, expr: Expr) -> DataFrameLike:
        """Uses expr and internal data to calculate the error described by expr"""
        function_data = dict()
        err_expr = self._error_func(expr=expr)
        for var in err_expr.free_symbols:
            var = str(var)
            series = self.data[var]
            function_data[var] = series.to_numpy()
        return self._expr_to_np(expr=err_expr)(**function_data)

    def _error_func(self, expr: Expr) -> Expr:
        """Uses expr and find its groessenunsicherheits methoden representation"""
        vs = list(ordered(expr.free_symbols))

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = Expr()
        # e_func = 0
        errs = " ".join([f"d{s}" for s in vs])
        er = simp.symbols(errs)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(expr, vs), er):
            e_func = e_func + abs(c) * s
        return e_func

    def find_min_sign_changes(self, df: DataFrameLike, identifier) -> DataFrameLike:
        """Operation on data column <identifier> find places where a sign flip happend"""
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
        return df.iloc[change_idx[range(len(change_idx)), min_idx]]  # type:ignore

    def probe_for_zeros(self, start, end, identifier, filter):
        # TODO remove t filter
        pz = self.find_min_sign_changes(self.data, identifier)[filter]
        pz = pz[pz.between(start, end)]
        print(pz)
        self.data = self.data[self.data[filter].between(start, end)]
        return pz

    def rms_wave(self, x, y):
        """Calculates the RMS value of x and y data"""
        I1 = integrate.simpson(y**2, x)
        rms = np.sqrt(I1 / (max(x) - min(x)))
        return rms

    def avr_wave(self, x, y):
        """Calculates the AVR value of x and y data"""
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

    def find_phase_and_gain(
        self, t: Sequence[int], v: Sequence[int], f: int
    ) -> tuple[int, int]:
        a_sum = 0  # now do it all again for the output wave
        b_sum = 0
        w = 2 * np.pi * f
        for time, voltage in zip(t, v):
            a_sum = a_sum + voltage * np.cos(w * time)
            b_sum = b_sum + voltage * np.sin(w * time)
        a = a_sum * 2 / len(t)
        b = b_sum * 2 / len(t)
        mag = np.sqrt(a**2 + b**2)
        phase = np.arctan2(a, b)
        print(mag)
        print(phase)
        return (mag, phase)

    def plot_function(
        self,
        axes: plt.Axes,
        x: str,
        y: str,
        expr: Expr,
        label: str,
        errors: bool = False,
        style: str = "r",
    ):
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
        self._set_x_y_label(axes, x, y)
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

    def plot_fit(
        self,
        axes: plt.Axes,
        x: str,
        y: str,
        eqn,
        style: str = "r",
        label: str = "fit",
        use_all_known: bool = False,
        offset: tuple[int, int] = (0, 0),
        guess: dict[str, int] | None = None,
        bounds: list[dict[str, str]] | None = None,
        add_fit_params: bool = True,
        granularity: int = 10000,
        gof: bool = False,
        sigmas: int = 1,
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
        axes = self._set_x_y_label(ax=axes, x=x, y=y)
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

    def savefig(self, name: str, clear: bool = True) -> Optional[plt.Axes]:
        """Use this method to save your figures"""
        self.figure.savefig(f"./Output/{self.name}/" + name, dpi=400)
        if clear:
            self.figure.clear()
            ax = self.figure.add_subplot()
            return ax

    def add_text(
        self, axes: plt.Axes, keyvalue: dict[str, Any], offset: tuple = (0, 0)
    ):
        """Write text inside on an axes"""
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