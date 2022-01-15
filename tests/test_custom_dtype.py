from labtool_ex2.dtype import *  # type: ignore
import pandas as pd
import numpy as np
import uncertainties as u
import uncertainties.unumpy as unp

test_list = list(range(10)) + [4, np.nan]  # type: ignore
test_uarray = unp.uarray(test_list, [0.2]*len(test_list))

def test1():
    ufloatarray = UfloatArray(test_list)  # type: ignore
    print(ufloatarray)

def test2():
    series = pd.Series(test_list, dtype="ufloat")
    print(series.u.n)
    
def test3():
    df_uarray = pd.DataFrame({"ufloats": UfloatArray(test_uarray), "ints": range(len(test_uarray)), })
    print([(x.nominal_value, x.std_dev, x.tag) for x in test_uarray])
    print(df_uarray.dtypes)
    print(type(df_uarray["ufloats"]))
    
def test4():
    print(pd.Series(UfloatArray(test_uarray)))
    
def test5():
    df_uarray = pd.DataFrame({"ufloats": test_uarray, "ints": range(len(test_uarray))})
    print(df_uarray.dtypes)


test2()