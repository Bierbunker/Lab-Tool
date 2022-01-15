import numpy as np
import pandas as pd
import uncertainties.unumpy as unp
import random
from labtool_ex2.uarray import UArray


uarr = UArray(unp.uarray(random.sample(range(0, 100), 10), list(random.sample(range(0, 100), 9) + [np.nan])))  # type: ignore
uarr = UArray(list(uarr) + [1, 2, 3])
print(uarr)
print(uarr.dtype)
print(uarr.n)
print(uarr.n.dtype)
print(uarr.s)


df = pd.DataFrame({"uarray": uarr})