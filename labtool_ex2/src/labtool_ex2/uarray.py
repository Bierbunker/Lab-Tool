from numpy import asarray, ndarray
from uncertainties import ufloat_fromstr


class UArray(ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if hasattr(input_array, "dtype"):
            # print(input_array.dtype)
            for val in input_array:
                if isinstance(val, str):
                    try:
                        input_array = [ufloat_fromstr(x) for x in input_array]
                    except ValueError:
                        print(f"Warning conversion failed for {input_array}")
                    break

        return asarray(input_array).view(cls)

    @property
    def n(self):
        return asarray([catch(lambda x: x.n, value) for value in self])

    # ufloat_from_str(x.__format__()).n

    @property
    def s(self):
        return asarray(
            [catch(lambda x: x.s, value, handle=lambda _: 0) for value in self]
        )


# https://stackoverflow.com/questions/1528237/how-to-handle-exceptions-in-a-list-comprehensions
def catch(func, arg, handle=lambda x: x):
    try:
        return func(arg)
    except AttributeError:
        return handle(arg)
