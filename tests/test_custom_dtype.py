import labtool_ex2
from labtool_ex2.dtype import UfloatArray
import pandas as pd
import numpy as np
import uncertainties.unumpy as unp
import inspect
import sys

test_list = list(range(10)) + [4, np.nan]  # type: ignore
test_uarray = unp.uarray(test_list, [0.2] * len(test_list))


def test1():
    print("\nTest 1\n")
    ufloatarray = UfloatArray(test_list)
    print(ufloatarray)


def test2():
    print("\nTest 2\n")
    series = pd.Series(test_uarray, dtype="ufloat")
    print(series)


def test3():
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


def test4():
    print("\nTest 4\n")
    print(type(pd.Series(UfloatArray(test_uarray)).dtype))


def test5():
    print("\nTest 5\n")
    df_uarray = pd.DataFrame({"ufloats": test_uarray, "ints": range(len(test_uarray))})
    print(df_uarray.dtypes)


def test6():
    print("\nTest 6\n")
    series = pd.Series(test_uarray, name="u", dtype="ufloat")
    print(series.u.s)


def test7():
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


def test8():
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


def test9():
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


def test10():
    print("\nTest 10\n")
    df = pd.read_csv("test_csv.csv")
    print(df)
    print(df.u.n)
    print(df.u.s)
    print(df.u.com)
    print(df.u.com.u.n)
    print(df.u.com.u.s)
    print(df.u.com.u.sep)
    print(df.astype("float"))


# UArray.n /.s (sub-class from np.ndarray)
# pd.Series.u.n /.s
# pd.DataFrame.u.n /.s

test10()

# for _, func in inspect.getmembers(sys.modules['__main__'], inspect.isfunction):
#    func()
