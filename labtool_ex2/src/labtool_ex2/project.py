import numpy as np

import pandas
import re
import io
import sympy as simp
import inspect
from uncertainties.unumpy import isnan
from collections.abc import Iterable
from pandas import DataFrame
from sympy.utilities.iterables import ordered
import matplotlib
from typing import Callable, Union, Any, Sequence, Optional
from sympy.core.expr import Expr
from scipy import integrate
from scipy.signal import find_peaks

# from lmfit import Parameters
# from lmfit import conf_interval2d
# from lmfit.models import LinearModel

from lmfit.models import ExpressionModel
from uncertainties import ufloat

# from uncertainties.core import format_num
from .helpers import round_up, split_into_args_kwargs, unique_std_devs

from pathlib import Path
from .LatexPrinter import latex
import labtool_ex2.symbol as s

# from .symbol import _custom_var

from sympy import Matrix, hessian
from sympy.interactive import printing
from matplotlib.legend import Legend

# dont fucking move this line

matplotlib.use("Agg")
from matplotlib.legend import _get_legend_handles_labels  # noqa
import matplotlib.pyplot as plt  # noqa
from .dtype import UfloatDtype


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
        self._err_prefix: str = "d"
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
            self._output_dir = f"./Output/{name}"
            self._tab_dir = f"./tables/"
            self._fig_dir = f"./figures/"
            tabp = Path(self._output_dir + self._tab_dir)
            figp = Path(self._output_dir + self._fig_dir)
            tabp.mkdir(parents=True, exist_ok=True)
            figp.mkdir(parents=True, exist_ok=True)
            p.mkdir(parents=True, exist_ok=True)

        # inject variables into global namespace can be improved if necessary however sympy does the same thing
        s._custom_var(list(global_variables), project=self)
        # simp.var(list(global_variables), cls=Symbol, df=self.data)

    def vload(self, names: Optional[list[str]] = list(), **args):
        if names:
            s._custom_var(names=names, project=self, **args)
        else:
            s._custom_var(names=list(self.gv), project=self, **args)

    @property
    def output_dir(self):
        """The output_dir property."""
        return self._output_dir

    @property
    def tab_dir(self):
        """The tab_dir property."""
        return self._tab_dir

    @property
    def fig_dir(self):
        """The fig_dir property."""
        return self._fig_dir

    @fig_dir.setter
    def fig_dir(self, value):
        self._fig_dir = value

    @tab_dir.setter
    def tab_dir(self, value):
        self._tab_dir = value

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def err_prefix(self) -> str:
        """The err_prefix property."""
        return self._err_prefix

    @err_prefix.setter
    def err_prefix(self, value):
        self._err_prefix = value

    def __str__(self) -> str:
        return f"""This is the {self.name} Project \n following mappings happen in this project {self.gm}"""

    def load_data(
        self,
        path: Union[str, Path],
        loadnew: bool = False,
        clean: bool = True,
    ) -> None:
        """
        path is the file containing csv data
        loadnew resets the currently loaded data dataframe should be used when importing many files
        clean should not be touched if you don't know what you are doing contact max if raw data is need"""
        print("\n\nLoading Data from: " + str(path))
        if loadnew:
            # self.data.drop(self.data.index, inplace=True)
            self.data = pandas.DataFrame(data=None)
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
            if not re.match(rf"^{self._err_of('')}(\w)+(\.\d+)?$", var):
                reg_var = rf"^{var}(\.\d+)?$"
                reg_err = rf"^{self._err_of('')}{var}(\.\d+)?$"
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
                    sem.name = self._err_of(var)
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

        # self.data.dropna(inplace=True)
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

    def plot_histo(
        self,
        axes: plt.Axes,
        x: Union[str, s.Symbol, DataFrameLike],
        label: str,
        bins: int = 0,
        offset: Union[tuple[int, int], list[int]] = (0, 0),
        *args,
        style: str = "r",
        **kwargs,
    ):
        """Plots the histogram of a pandas series and draws a fit

        :ser: TODO
        :bins: TODO
        :name: TODO
        :returns: TODO

        """
        if isinstance(x, Expr):
            x_name = self._infer_name(kw="x", pos=1)
            raw_x: pandas.Series = self.data[x_name]
        else:
            raw_x, x_name = self._parse_input(x)

        tempv = None
        tempm = None
        p = "p"
        if p in self.gv:
            tempv = self.gv[p]

        if p in self.gm:
            tempm = self.gm[p]
        self.gv[p] = "1"
        self.gm[p] = p
        axes = self._set_x_y_label(ax=axes, x=x_name, y=p, color=style)
        del self.gv[p]
        del self.gm[p]
        if tempv:
            self.gv[p] = tempv
        if tempm:
            self.gm[p] = tempm

        counts, bins = np.histogram(raw_x, bins)
        bin_widths = np.diff(bins)
        mu = ufloat(raw_x.mean(), raw_x.sem())
        sigma = ufloat(raw_x.std(), 0)
        raw_x.hist(ax=axes, bins=bins, density=True)
        val = np.linspace(raw_x.min(), raw_x.max(), 1000)
        p = np.exp(-0.5 * ((val - mu.n) / sigma.n) ** 2) / (
            np.sqrt(2 * np.pi) * sigma.n
        )
        axes.plot(
            val, p, *args, linestyle="-", color=style, label=label + " fit", **kwargs
        )

        paramstr = rf"$\mu = {mu}$ " + self.gv[x_name] + "\n"
        paramstr += rf"$\sigma = {round_up(sigma.n,4)}$ " + self.gv[x_name] + "\n"
        paramstr += rf"$N = {raw_x.size}$ " + "1"
        if np.allclose(bin_widths[0], bin_widths):
            paramstr += "\n" + rf"$h = {round_up(bin_widths[0],4)}$ " + self.gv[x_name]

        self.add_text(axes=axes, text=paramstr, offset=offset, color=style)

    """
    TODO write own to latex function export data with the use of siunitx plugin
    """

    def print_expr(self, expr: Expr):
        """Prints the expression in latex format and substitutes the variables according to the mapping"""
        latex(expr=expr, symbol_names=self.gm)

    def _unit_of(self, sl: str) -> str:
        if sl.startswith(self.err_prefix):
            unit = self.gv[sl.removeprefix(self.err_prefix)]
        else:
            unit = self.gv[sl]
        return unit

    def _table_format(
        self, ser: pandas.Series, split: bool
    ) -> Union[tuple[str, str], str]:
        exp = 0
        pre_dot = 0
        post_dot = 0
        e_pre_dot = 0
        e_post_dot = 0
        if split:
            for _, val in ser.items():
                fmt_x = val.__format__("")
                if "e" in fmt_x:
                    fmt_x, _exp = fmt_x.split("e", 1)
                    if len(_exp.strip()) > exp:
                        exp = len(_exp.strip())

                nom, err = fmt_x.replace("(", "").replace(")", "").split("+/-", 1)
                m = re.match(r"(?P<pre>\d+)(\.(?P<post>\d+))*", nom)
                if m:
                    m = m.groupdict()
                    _pre_dot = m["pre"]
                    _post_dot = m["post"]
                    if _pre_dot and len(_pre_dot) > pre_dot:
                        pre_dot = len(_pre_dot)
                    if _post_dot is not None and len(_post_dot) > post_dot:
                        post_dot = len(_post_dot)
                m = re.match(r"(?P<pre>\d+)(\.(?P<post>\d+))*", err)
                if m:
                    m = m.groupdict()
                    _e_pre_dot = m["pre"]
                    _e_post_dot = m["post"]
                    if _e_pre_dot and len(_e_pre_dot) > e_pre_dot:
                        e_pre_dot = len(_e_pre_dot)
                    if _e_post_dot is not None and len(_e_post_dot) > e_post_dot:
                        e_post_dot = len(_e_post_dot)
        else:
            for _, val in ser.items():
                fmt_x = val.__format__("S")
                if "e" in fmt_x:
                    fmt_x, _exp = fmt_x.split("e", 1)
                    if len(_exp.strip()) > exp:
                        exp = len(_exp.strip())

                m = re.match(
                    r"(?P<pre>\d+)(\.(?P<post>\d+))*\((?P<err>(\d|\.)+)\)", fmt_x
                )
                if m:
                    m = m.groupdict()
                    _pre_dot = m["pre"]
                    _post_dot = m["post"]
                    _e_post_dot = m["err"]
                    if _pre_dot and len(_pre_dot) > pre_dot:
                        pre_dot = len(_pre_dot)
                    if _post_dot and len(_post_dot) > post_dot:
                        post_dot = len(_post_dot)
                    if _e_post_dot and len(_e_post_dot) > e_post_dot:
                        e_post_dot = len(_e_post_dot)
        if split:
            if exp:
                return f"{pre_dot}.{post_dot}e{exp}", f"{e_pre_dot}.{e_post_dot}e{exp}"
            return f"{pre_dot}.{post_dot}", f"{e_pre_dot}.{e_post_dot}"
        else:
            if exp:
                return f"{pre_dot}.{post_dot}({e_post_dot})e{exp}"
            return f"{pre_dot}.{post_dot}({e_post_dot})"

    def print_table_expr(
        self,
        expr: Expr,
        name: str = "",
        **kwargs,
    ):
        self.print_table(*expr.free_symbols, name=name, **kwargs)

    def print_table(
        self,
        *args: Union[str, s.Symbol, s.Basic],
        name: str = "",
        split: bool = False,
        inline_units: bool = False,
        filter_all_same: bool = False,
        bold: bool = True,
        options: str = "",
        vars: Optional[list[str] | list[s.Symbol]] = list(),
        censor: Optional[list] = [np.nan, float("nan")],
    ):
        """If you want to print a latex table this is the function you need.
        With split you specify if the error should be printed separately from the nominal value.
        With inline_units you specify if the units of the variables should be printed on the next line or inline.
        With name you customize the name of the output file
        With vars is an alternative way of passing values to this function print_table(*vars) is preferred
        """
        cols = list()
        for arg in args:
            if isinstance(arg, str):
                cols.append(arg)
            elif isinstance(arg, s.Symbol):
                cols.append(arg.name)
            else:
                raise Exception("Unsupported type in args")
        if vars:
            for arg in vars:
                if isinstance(arg, str):
                    cols.append(arg)
                elif isinstance(arg, s.Symbol):
                    cols.append(arg.name)
                else:
                    raise Exception("Unsupported type in args")

        # first handle if split or not and check if ufloat arrays
        # self.data = self.data.astype("ufloat")

        df = self.data.u.com
        # print(df)
        df = df[cols].astype("ufloat")
        unique_devs = unique_std_devs(df)
        if not filter_all_same:
            unique_devs = [False] * len(unique_devs)

        if split:
            coldict = self._col_rename(df.u.sep.columns)
            colcp = coldict.copy()
            for i, (key, val) in enumerate(colcp.items()):
                if i % 2 == 1 and unique_devs[i // 2]:
                    del coldict[key]

        else:
            coldict = self._col_rename(df.columns)

        if inline_units:
            header = (
                " & ".join(
                    [
                        self._tblr_esc(val + " / " + self._unit_of(key), bold)
                        for key, val in coldict.items()
                    ]
                )
                + "\\\\\n"
            )
        else:
            header = (
                " & ".join([self._tblr_esc(val, bold) for _, val in coldict.items()])
                + "\\\\\n"
            )
            header += (
                " & ".join(
                    [
                        self._tblr_esc(self._unit_of(key), bold)
                        for key, _ in coldict.items()
                    ]
                )
                + "\\\\\n"
            )

        numRows = df.shape[0]
        colspec = []
        for i, (_, ser) in enumerate(df.items()):
            if split:
                nom, err = self._table_format(ser, split)
                colspec.append(f"S[table-format={nom}]")
                if not unique_devs[i]:
                    colspec.append(f"S[table-format={err}]")
            else:
                val = self._table_format(ser, split)
                if unique_devs[i]:
                    colspec.append(f"S[table-format={val.split('(',1)[0]}]")
                else:
                    colspec.append(f"S[table-format={val}]")

        begin = "\\begin{tblr}{" + options + "colspec={" + "".join(colspec) + "}}\n"
        end = "\\end{tblr}\n"

        output = io.StringIO()
        output.write(begin)
        output.write(header)
        if split:

            def format_value(x, i):  # type: ignore
                fmt_x = x.__format__("")
                if "e" in fmt_x:
                    val, exp = fmt_x.split("e", 1)
                    if unique_devs[i]:
                        val = val.split("+/-", 1)[0] + "e" + exp
                    else:
                        val = (
                            val.replace("+/-", "e" + exp + " & ")
                            .replace("(", "")
                            .replace(")", "")
                            + "e"
                            + exp
                        )
                else:
                    if unique_devs[i]:
                        val = fmt_x.split("+/-", 1)[0]
                    else:
                        val = fmt_x.replace("+/-", " & ")
                return val

        else:

            def format_value(x, i):
                try:
                    fmt_x = x.__format__("S")
                    if unique_devs[i]:
                        s = re.sub(r"\((.*?)\)", "", fmt_x)
                    else:
                        s = re.sub(
                            r"\((.*?)\)", lambda g: re.sub(r"\.", "", g[0]), fmt_x
                        )
                except Exception:
                    s = "{{{-}}}"
                return s

        for i in range(numRows):
            output.write(
                " & ".join(
                    [
                        format_value(val, j)
                        if val not in censor and not isnan(val)
                        else "{{{-}}}"
                        for j, val in enumerate(df.iloc[i])
                    ]
                )
                + "\\\\\n"
            )
        output.write(end)

        # output.seek(0)
        # print(output.read())
        if name:
            with open(
                self.output_dir + self.tab_dir + f"messreihe_{name}.tex", "w"
            ) as tf:
                tf.write(output.getvalue())
        else:
            with open(
                self.output_dir + self.tab_dir + f"messreihe_{df.name}.tex", "w"
            ) as tf:
                tf.write(output.getvalue())

    def _tblr_esc(self, s: str, bold: bool):
        if bold:
            s = r"\textbf{" + s + r"}"
        return f"{{{{{{{s}}}}}}}"

    def _col_rename(self, columns: list[str]) -> dict[str, str]:
        err_prefix = self.err_prefix
        rcol = dict()
        for colname in columns:
            if (
                colname.startswith(err_prefix)
                and colname.removeprefix(err_prefix) in self.gm
            ):
                rmcol = colname.removeprefix(err_prefix)
                coln = self.gm[rmcol]
                rcol[colname] = r"$\Delta " + coln + "$"
                continue
            if colname in self.gm:
                coln = self.gm[colname]
                rcol[colname] = "$" + coln + "$"
                continue
            if colname.startswith(err_prefix) and "\\" not in colname:
                subbed = colname.replace(err_prefix, "\\Delta ")
                rcol[colname] = "$" + subbed + "$"
        return rcol

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
                # print(f"testing {ax.get_xlabel()} and {ax.get_ylabel()}")
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

    def _add_error(
        self,
        axes: plt.Axes,
        raw_x: pandas.Series,
        raw_y: pandas.Series,
        x_name: str,
        y_name: str,
        style: str,
    ):
        errs = dict()
        if isinstance(raw_x.dtype, UfloatDtype):
            errs["xerr"] = raw_x.u.s
        else:
            try:
                name = self._err_of(x_name)
                errs["xerr"] = self.data[name].values
            except Exception as e:
                print(f"No xerr for {x_name} found")

        if isinstance(raw_y.dtype, UfloatDtype):
            errs["xerr"] = raw_y.u.s
        else:
            try:
                name = self._err_of(y_name)
                errs["yerr"] = self.data[name].values
            except Exception as e:
                print(f"No yerr for {y_name} found")
        axes.errorbar(raw_x, raw_y, fmt="none", capsize=3, color=style, **errs)

    def _parse_input(
        self,
        x: Union[str, s.Symbol, DataFrameLike],
    ) -> tuple[pandas.Series, str]:
        if isinstance(x, s.Symbol):
            raw_x = x.data
            x_name = x.name
        elif isinstance(x, str):
            raw_x = self.data[x]
            x_name = x
        elif isinstance(x, pandas.Series):
            raw_x = x
            x_name = x.name
        # elif issubclass(x,Expr):
        #     self._infer_name(kw=)
        else:
            raise Exception("Unsupported type")
        return (raw_x, x_name)  # type: ignore

    def _err_of(self, x: Union[str, s.Symbol, DataFrameLike]) -> str:
        if isinstance(x, s.Symbol):
            name = x.name
        elif isinstance(x, str):
            name = x
        elif isinstance(x, pandas.Series):
            name = x.name
        else:
            raise Exception("Unsupported type")
        return f"{self.err_prefix}{name}"

    def _expr_to_np(self, expr: Expr) -> Callable:
        """Converts a Sympy Expression to a numpy calculable function"""
        # print(tuple(expr.free_symbols))
        return simp.lambdify(tuple(expr.free_symbols), expr, "numpy")

    # maybe rename to ingest instead of inject
    def inject_err(self, expr: Expr, name: Optional[str] = None):
        """This function is only for readability of the finished code"""
        name = self._infer_name(name=name, kw="expr", pos=0)  # type: ignore
        self.resolve(expr, name, iserr=True)

    def _retrieve_params(self, expr: Expr):
        notfound = list()
        function_data = dict()
        err_function_data = dict()
        for var in expr.free_symbols:
            if isinstance(var, s.Symbol):
                if var.name not in self.data.columns:
                    notfound.append(var.name)
                    continue
                if not isinstance(var.data.dtype, UfloatDtype):
                    try:
                        err_function_data[self._err_of(var.name)] = self.data[
                            self._err_of(var.name)
                        ].to_numpy()  # type:ignore
                        err_function_data[var.name] = var.data.to_numpy()  # type:ignore
                    except KeyError:
                        pass

                function_data[var.name] = var.data.to_numpy()  # type:ignore
            else:
                print("ohno")

        if notfound:
            raise Exception(f"Data for the following variables are missing: {notfound}")
        return function_data, err_function_data

    # should be probably move to helpers
    def _infer_name(
        self,
        name: Optional[str] = None,
        kw: Optional[str] = None,
        pos: Optional[int] = None,
    ) -> str:
        if name is not None:
            return name

        inferred_name = ""

        from inspect import currentframe

        func_name = currentframe().f_back.f_code.co_name  # type: ignore
        # pint(dir(currentframe().f_back.f_code))
        # print(func_name)
        # line_of_code = inspect.stack()[2][4][0]  # type: ignore
        import linecache

        frame = currentframe().f_back.f_back  # type: ignore
        lineno = frame.f_lineno  # type: ignore
        co = frame.f_code  # type: ignore
        filename = co.co_filename
        name = co.co_name
        complete_function = ""
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, frame.f_globals)  # type: ignore
        complete_function = complete_function + line.split("#")[0].strip()
        while ")" not in line:
            lineno = lineno + 1
            line = linecache.getline(filename, lineno, frame.f_globals)  # type: ignore
            if not line.strip().startswith("#"):
                reg = r"(?P<params>(([^\W0-9]\w*)+(=(.)+)?,)+)"
                match = re.search(reg, line)
                if match is not None:
                    complete_function = (
                        complete_function + match.group("params").strip()
                    )

        if ")" not in complete_function:
            complete_function = complete_function + ")"

        # complete_function = complete_function + line.split("#")[0].strip()
        # if not line.strip().startswith("#"):
        #     reg = r"(?P<params>(([^\W0-9]\w*)+(=(.)+)?,)+)"
        #     match = re.search(reg, line)
        #     if match is not None:
        #         complete_function = complete_function + match.group("params").strip()
        # print(complete_function)
        # retVal =  '\n  File "%s", line %d, in %s' % (filename, lineno, name)
        # if line:
        #     retVal += "\n        " + line.strip()
        # return retVal
        reg = func_name + r"\((?P<params>.+)\)"
        match = re.search(reg, complete_function)
        if match is not None:
            # params = match.group("params").split(",")  # type: ignore
            # nargs, kwargs = eval(f'_args({match.group("params")})')
            nargs, kwargs = split_into_args_kwargs(match.group("params"))
            # print(nargs)
            # print(kwargs)
            if pos is not None and len(nargs) > pos:
                inferred_name = nargs[pos]  # type: ignore
            if kw is not None:
                try:
                    inferred_name = kwargs[kw]  # type: ignore
                except KeyError:
                    pass
            if not inferred_name.strip():
                raise Exception(
                    "You need to either pass pos or kw or both to the function"
                )
        else:
            raise Exception("MFQR do please call the function properly")
        return inferred_name

    # Maybe rename it to derive
    def resolve(self, expr: Expr, name: Optional[str] = None, iserr: bool = False):
        """As in '(of something seen at a distance) turn into a different form when seen more clearly:
        ex. the orange light resolved itself into four roadwork lanterns'"""

        new_var = self._infer_name(name=name, kw="expr", pos=0)  # type: ignore

        function_data, err_function_data = self._retrieve_params(expr=expr)

        self.data[new_var] = self._expr_to_np(expr=expr)(**function_data)
        if not iserr:
            if err_function_data:
                err_expr = self._error_func(expr=expr)
                err_function_data = {
                    sym.name: err_function_data[sym.name]  # type: ignore
                    for sym in err_expr.free_symbols
                }
                self.data[self._err_of(new_var)] = self._expr_to_np(expr=err_expr)(
                    **err_function_data
                )

        return s._custom_var([new_var], project=self)
        # TODO use physipy for unit translation and calculation so that units are automatically calculated

        # do = [var.name in self.data.columns for var in expr.free_symbols]
        # names = [var.name for var in expr.free_symbols]
        # if all(name in self.data.columns for name in names):
        #     print(names)

    def apply_df(
        self,
        expr: Expr,
        name: str | None = None,
        errors: bool = False,
    ) -> DataFrameLike:
        """Uses expr and internal data to calculate the value described by expr"""
        new_var = self._infer_name(name=name, kw="expr", pos=0)  # type: ignore

        function_data, _ = self._retrieve_params(expr=expr)

        self.data[new_var] = self._expr_to_np(expr)(**function_data)
        # TODO Add or reset definition of variable using f_locals no like sympy does
        # simp.var(new_var)
        return self.data[new_var]

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
        return self._groessen_propagation(expr=expr)

    def _groessen_propagation(self, expr: Expr) -> Expr:
        vs = list(ordered(expr.free_symbols))

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0  # type: ignore
        errs = " ".join([self._err_of(s) for s in vs])
        er = simp.symbols(errs)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(expr, vs), er):
            e_func = e_func + abs(c) * s  # type: Expr

        return e_func

    def _gauss_propagation(self, expr: Expr) -> Expr:
        vs = list(ordered(expr.free_symbols))

        def gradient(f, vs):
            return Matrix([f]).jacobian(vs)

        e_func = 0  # type: ignore
        errs = " ".join([self._err_of(s) for s in vs])
        er = simp.symbols(errs)
        if not isinstance(er, Iterable):
            er = [er]
        for c, s in zip(gradient(expr, vs), er):
            e_func = e_func + c**2 * s  # type: Expr

        return e_func

    # TODO load error function

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
        # print(pz)
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

    def plot_data(
        self,
        axes: plt.Axes,
        x: Union[str, s.Symbol, DataFrameLike],
        y: Union[str, s.Symbol, DataFrameLike],
        label: str,
        *args,
        style: str = "r",
        errors: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(x, Expr):
            x_name = self._infer_name(kw="x", pos=1)
            raw_x = self.data[x_name]
        else:
            raw_x, x_name = self._parse_input(x)
        if isinstance(y, Expr):
            y_name = self._infer_name(kw="y", pos=2)
            raw_y = self.data[y_name]
        else:
            raw_y, y_name = self._parse_input(y)
        axes = self._set_x_y_label(ax=axes, x=x_name, y=y_name, color=style)

        if errors:
            self._add_error(
                axes=axes,
                raw_x=raw_x,
                raw_y=raw_y,
                x_name=x_name,
                y_name=y_name,
                style=style,
            )

        axes.scatter(
            raw_x, raw_y, c=style, marker=".", s=39.0, label=label, *args, **kwargs
        )

    def plot_function(
        self,
        axes: plt.Axes,
        x: Union[str, s.Symbol, DataFrameLike],
        expr: Expr,
        label: str,
        y: Optional[Union[str, s.Symbol, DataFrameLike]] = None,
        style: str = "r",
        errors: bool = False,
        *args,
        **kwargs,
    ):

        if isinstance(x, Expr):
            x_name = self._infer_name(kw="x", pos=1)
            raw_x = self.data[x_name]
        else:
            raw_x, x_name = self._parse_input(x)
        if y is not None:
            raw_y, y_name = self._parse_input(y)
        else:
            y_name = self._infer_name(name=None, kw="expr", pos=2)  # type: ignore

        axes = self._set_x_y_label(ax=axes, x=x_name, y=y_name, color=style)

        raw_y = self.apply_df(expr)

        if errors:
            self._add_error(
                axes=axes,
                raw_x=raw_x,
                raw_y=raw_y,
                x_name=x_name,
                y_name=y_name,
                color=style,
            )

        axes.plot(
            raw_x, raw_y, linestyle="-", color=style, label=label, *args, **kwargs
        )
        # if errors:
        #     if self._err_of(y) in self.data:
        #         dely = self.data[self._err_of(y)]
        #         axes.fill_between(
        #             raw_x, raw_y - dely, raw_y + dely, color=style, alpha=0.4
        #         )

    def plot_fit(
        self,
        axes: plt.Axes,
        x: Union[str, s.Symbol, DataFrameLike],
        y: Union[str, s.Symbol, DataFrameLike, Expr],
        eqn,
        style: str = "r",
        label: str = "fit",
        use_all_known: bool = False,
        offset: Union[tuple[int, int], list[int]] = (0, 0),
        guess: dict[str, Union[int, float]] | None = None,
        bounds: list[dict[str, str | float | int]] | None = None,
        add_fit_params: bool = True,
        granularity: int = 10000,
        gof: bool = False,
        sigmas: int = 1,
        scale_covar: bool = False,
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
        if isinstance(x, Expr):
            x_name = self._infer_name(kw="x", pos=1)
            raw_x = self.data[x_name]
        else:
            raw_x, x_name = self._parse_input(x)
        if isinstance(y, Expr):
            y_name = self._infer_name(kw="y", pos=2)
            raw_y = self.data[y_name]
        else:
            raw_y, y_name = self._parse_input(y)
        x_continues_plot_data = np.linspace(min(raw_x), max(raw_x), granularity)
        y_fit = raw_y.values.reshape(1, -1)  # type: ignore
        independent_vars = dict()
        cont_independent_vars = dict()
        if hasattr(eqn, "lhs"):
            eqn = eqn[1]
        if use_all_known:
            vars = [str(sym) for sym in eqn.free_symbols if str(sym) != y_name]
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
                independent_vars=[x_name],
            )
            x_fit = raw_x.values.reshape(1, -1)  # type: ignore
            independent_vars[x_name] = x_fit
            cont_independent_vars[x_name] = np.linspace(
                min(raw_x), max(raw_x), granularity
            )

        if bounds:
            for bound in bounds:
                mod.set_param_hint(
                    name=bound["name"], min=bound["min"], max=bound["max"]
                )
                mod.print_param_hints()

        if guess:
            pars = mod.make_params(**guess)
        else:
            # pars = mod.guess(raw_y, x_name=raw_x)
            # TODO custom error needs to be implemented
            print("No guess was passed")
            raise Exception
        # print(independent_vars)
        # print(pars)

        if self._err_of(y_name) in self.data:
            weights = 1 / (self.data[self._err_of(y_name)])
            weights = weights.values.reshape(1, -1)
            out = mod.fit(
                y_fit,
                pars,
                weights=weights,
                scale_covar=scale_covar,
                **independent_vars,
            )
        else:
            # out = mod.fit(y_fit, pars,scale_covar=False, **independent_vars)
            out = mod.fit(y_fit, pars, **independent_vars)
        axes = self._set_x_y_label(ax=axes, x=x_name, y=y_name)
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

            self.add_text(axes=axes, text=paramstr, offset=offset, color=style)

        resp = np.array(out.eval(**cont_independent_vars))
        # print(resp)
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
        dely = out.eval_uncertainty(**cont_independent_vars, sigma=sigmas)
        # axes.plot(x, data)
        # axes.plot(x, out.best_fit)
        axes.fill_between(
            x_continues_plot_data, resp - dely, resp + dely, color=style, alpha=0.4
        )
        axes.plot(x_continues_plot_data, resp, style, label=f"{label} fit")
        print(out.fit_report(min_correl=0.25))
        return out.params

    def savefig(self, name: str, clear: bool = True) -> Optional[plt.Axes]:
        """Use this method to save your figures"""
        self.figure.savefig(self.output_dir + self.fig_dir + name, dpi=400)
        if clear:
            self.figure.clear()
            ax = self.figure.add_subplot()
            return ax

    def add_text(
        self,
        axes: plt.Axes,
        keyvalue: Optional[dict[str, Any]] = None,
        text: Optional[str] = None,
        offset: Union[tuple[int, int], list[int]] = (0, 0),
        color="#616161",
    ):
        """Adds parameters as text on an axes"""
        # maxes = max([_.zorder for _ in axes.get_children()])
        # print(maxes)
        maxes = axes
        xmin, xmax = maxes.get_xlim()
        ymin, ymax = maxes.get_ylim()
        if keyvalue is not None:
            for i, (name, param) in enumerate(keyvalue.items()):
                uparam = param
                maxes.text(
                    s=rf"${self.gm[name]} = {uparam}$ " + self.gv[name],
                    bbox={"facecolor": color, "alpha": 0.85},
                    x=(xmax - xmin) / 100 * (5 + offset[0]) + xmin,
                    y=(ymax - ymin) * (offset[1] / 100 + (i + 1) / 7) + ymin,
                    fontsize=10,
                )
        if text is not None:
            maxes.text(
                s=text,
                bbox={"facecolor": color, "alpha": 0.85},
                x=(xmax - xmin) / 100 * (5 + offset[0]) + xmin,
                y=(ymax - ymin) * (offset[1] / 100 + 1 / 7) + ymin,
                fontsize=10,
            )
