from labtool_ex2.dtype import UfloatArray
from labtool_ex2.dtype import UfloatDtype
from labtool_ex2.uarray import UArray
from uncertainties.core import AffineScalarFunc, Variable

import pandas as pd
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import unumpy
import os

test_list = list(range(10)) + [4, np.nan]  # type: ignore
test_uarray = unp.uarray(test_list, [0.2] * len(test_list))


def test_1():
    print("\nTest 1\n")
    ufloatarray = UfloatArray(test_list)  # type: ignore
    print(ufloatarray)


def test_2():
    print("\nTest 2\n")
    series = pd.Series(test_uarray, dtype="ufloat")
    print(series)
    print(series.dtype)
    print(type(series[0]))


def test_3():
    print("\nTest 3\n")
    df_uarray = pd.DataFrame(
        {
            "ufloats": UfloatArray(test_uarray),
            "ints": range(len(test_uarray)),
        }
    )
    print([(x.nominal_value, x.std_dev, x.tag) for x in test_uarray])
    print(df_uarray.dtypes)
    print(type(df_uarray["ufloats"]))


def test_4():
    print("\nTest 4\n")
    assert isinstance(pd.Series(UfloatArray(test_uarray)).dtype, UfloatDtype)
    assert (
        str(type(pd.Series(UfloatArray(test_uarray)).dtype))
        == "<class 'labtool_ex2.dtype.UfloatDtype'>"
    )


def test_5():
    print("\nTest 5\n")
    df_uarray = pd.DataFrame({"ufloats": test_uarray, "ints": range(len(test_uarray))})
    print(df_uarray.dtypes)


def test_6():
    print("\nTest 6\n")
    series = pd.Series(test_uarray, name="u", dtype="ufloat")
    print(series.u.s)
    assert issubclass(type(series.iloc[0]), AffineScalarFunc)


def test_7():
    print("\nTest 7\n")
    ints = range(len(test_uarray))
    df_uarray = pd.DataFrame(
        {
            "ufloats": test_uarray,
            "ints": ints,
            "strings": [chr(num**2) for num in ints],
        }
    )
    print(f"normal\n{df_uarray}\n")
    print(f"n\n{df_uarray.u.n}\n")
    print(f"s\n{df_uarray.u.s}\n")
    print(f"dtypes\n{df_uarray.dtypes}\n")
    print(f"dtypes n\n{df_uarray.u.n.dtypes}\n")


def test_8():
    print("\nTest 8\n")
    ints = range(len(test_uarray))
    df_uarray = pd.DataFrame(
        {
            "ufloats": test_uarray,
            "ints": ints,
            "strings": [chr(num**2) for num in ints],
        }
    )
    df_4 = df_uarray * 4
    print(df_uarray)
    print((df_uarray * 4).iloc[1:-1, :].u.sep)
    print(type(df_4.iloc[0, 0]))


def test_9():
    print("\nTest 9\n")
    ints = range(len(test_uarray))
    df_uarray = pd.DataFrame(
        {
            "ufloats": test_uarray,
            "ints": ints,
            "strings": [chr(num**2) for num in ints],
        }
    )
    print(df_uarray.u.sep)
    print(df_uarray.u.sep.u.com)
    print(f"\nfor comparison:\n{df_uarray}")


def test_10():
    print("\nTest 10\n")
    path = os.path.join(os.path.dirname(__file__), "data/input/test_csv.csv")
    df = pd.read_csv(str(path))
    # print(df.u.n)  # type: ignore
    assert all(df[["P1", "U", "U1", "I1", "U2", "I2", "Ur", "xl"]] == df.u.com.u.n)  # type: ignore
    assert all(
        df[["dP1", "dU", "dU1", "dI1", "dU2", "dI2", "dUr", "dxl"]].rename(  # type: ignore
            lambda name: name[1:], axis="columns"
        )
        == df.u.com.u.s  # type: ignore
    )
    # assert all(df[["P1", "U", "U1", "I1", "U2", "I2", "Ur", "xl"]] != df.u.n)
    assert all((df.u.s == 0).all())  # type: ignore
    # print(df.u.com)  # type: ignorefrom pathlib import Path
    # path = os.path.join(os.path.dirname(__file__), "data/input/test_csv.csv")
    filepath = os.path.join(os.path.dirname(__file__), "./data/output/df.u.com.csv")
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # temp = pd.read_csv(filepath).astype("ufloat")  # type: ignore
    assert all(df.u.com == pd.read_csv(filepath).astype("ufloat"))  # type: ignore
    filepath = os.path.join(os.path.dirname(__file__), "./data/output/df.u.com.u.n.csv")
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # df.u.com.u.n.read_csv(filepath)
    assert all(df.u.com.u.n == pd.read_csv(filepath))  # type: ignore
    filepath = os.path.join(os.path.dirname(__file__), "./data/output/df.u.com.u.s.csv")
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # df.u.com.u.s.to_csv(filepath)
    assert all(df.u.com.u.s == pd.read_csv(filepath))  # type: ignore
    filepath = os.path.join(
        os.path.dirname(__file__), "./data/output/df.u.com.u.sep.csv"
    )
    assert all(df.u.com.u.sep == pd.read_csv(filepath))  # type: ignore
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # df.u.com.u.sep.to_csv(filepath)
    # print(df.u.com.u.n)  # type: ignore
    # print(df.u.com.u.s)  # type: ignore
    # print(df.u.com.u.sep)  # type: ignore
    # test if it is idempotent
    assert (
        len(df.u.com.u.sep.columns) == len(df.columns)  # type: ignore
        and len(df.u.com.u.sep.columns) == 2 * len(df.u.com.columns)  # type: ignore
        and len(df.columns) == 2 * len(df.u.com.columns)  # type: ignore
    )
    # print(df.astype("float"))  # type: ignore


def test_11():
    print("\nTest 11\n")
    path = os.path.join(os.path.dirname(__file__), "data/input/test_csv.csv")
    df = pd.read_csv(str(path))
    df = df.u.com  # type: ignore
    df = df.astype("ufloat")
    assert all(isinstance(dtype, UfloatDtype) for dtype in df.dtypes)
    assert all("float64" == dtype for dtype in df.u.sep.dtypes)
    assert all(isinstance(dtype, UfloatDtype) for dtype in df.u.sep.u.com.dtypes)


def test_12():
    print("\nTest 12\n")
    path = os.path.join(os.path.dirname(__file__), "data/input/test_csv.csv")
    df = pd.read_csv(str(path))
    assert all(df == df.u.com.u.sep)  # type: ignore


def test_13():
    print("\nTest 13\n")
    path = os.path.join(os.path.dirname(__file__), "data/input/test_csv.csv")
    df = pd.read_csv(str(path))
    df = df.u.com  # type: ignore
    y = df.P1 + df.U.astype("ufloat")
    assert type(y[0]) == AffineScalarFunc


def test_14():
    """
    This tests the automatic conversion of strings to ufloat
    """
    filepath = os.path.join(os.path.dirname(__file__), "./data/output/df.u.com.csv")
    temp = pd.read_csv(filepath).astype("ufloat")  # type: ignore
    # temp = pd.read_csv(filepath, dtype=UfloatDtype()).astype(UfloatDtype())  # type: ignore
    temp.u.sep
    P1 = unumpy.uarray(temp.P1.u.n, temp.P1.u.s)
    U = unumpy.uarray(temp.U.u.n, temp.U.u.s)
    assert all(temp.dtypes == "ufloat")

    # print(pd.to_numeric(temp.P1))
    temp["y"] = temp.P1 + temp.U
    # print(P1 + U)
    # print(temp.y)
    # for x, y in zip(temp.y, P1 + U):
    #     print(x.n, x.s, y.n, y.s)
    #     print(type(x), type(y))
    #     print(x == y, x.n == y.n, x.s == y.s)
    dimsum = UArray(P1 + U)
    assert all(
        temp.y != dimsum
    )  # I know this is stupid however uncertainties does it this way
    assert all(temp.y.u.n == dimsum.n)
    assert all(temp.y.u.s == dimsum.s)
    assert type(temp.y[0]) == AffineScalarFunc
    assert type(temp.iloc[0, 0]) == Variable


# should raise error
# def test_15():
#     series = pd.Series(test_uarray, dtype="ufloat")
#     try:
#         series.u.sep
#     except:


# UArray.n /.s (sub-class from np.ndarray)
# pd.Series.u.n /.s
# pd.DataFrame.u.n /.s


# for _, func in inspect.getmembers(sys.modules['__main__'], inspect.isfunction):
#    func()
# test_14()
