from sympy.printing.latex import LatexPrinter


class LabLatexPrinter(LatexPrinter):
    """Print derivative of a function of symbols in a shorter form."""

    def _print_Symbol(self, expr, style="plain"):
        if str(expr) in self._settings["symbol_names"]:
            return self._settings["symbol_names"][str(expr)]

        return self._deal_with_super_sub(expr.name, style=style)


def latex(expr, **settings):
    """Most of the printers define their own wrappers for print().
    These wrappers usually take printer settings. Our printer does not have
    any settings.
    """
    print(LabLatexPrinter(settings=settings).doprint(expr))
