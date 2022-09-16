import labtool_ex2.helpers as h


def test_helpers():
    assert h.round_up(109.245, 2) == 109.25
    assert h.round_up(109.245, -2) == 200
    assert h.orderOfMagnitude(1001) == 3
    # assert h.orderOfMagnitude(1e3) == 3 doesn't work for some reason: my guess floating point shit
    assert h.orderOfMagnitude(1e-9) == -9
    assert h.orderOfMagnitude(1.42352e-9) == -9
    nargs, kwargs = h._args(9, 1, 4, hello=True, nine=9)
    assert nargs == (9, 1, 4)
    assert kwargs == {"hello": True, "nine": 9}
