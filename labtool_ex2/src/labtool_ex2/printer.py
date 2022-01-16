class Printer:
    """Generic printer
    Its job is to provide infrastructure for implementing new printers easily.
    If you want to define your custom Printer or your custom printing method
    for your custom class then see the example above: printer_example_ .
    """

    _global_settings = {}  # type: Dict[str, Any]

    _default_settings = {}  # type: Dict[str, Any]

    printmethod = None  # type: str

    @classmethod
    def _get_initial_settings(cls):
        settings = cls._default_settings.copy()
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val
        return settings

    def __init__(self, settings=None):
        self._str = str

        self._settings = self._get_initial_settings()
        self._context = dict()  # mutable during printing

        if settings is not None:
            self._settings.update(settings)

            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)

        # _print_level is the number of times self._print() was recursively
        # called. See StrPrinter._print_Float() for an example of usage
        self._print_level = 0

    @classmethod
    def set_global_settings(cls, **settings):
        """Set system-wide printing settings."""
        for key, val in settings.items():
            if val is not None:
                cls._global_settings[key] = val

    @property
    def order(self):
        if "order" in self._settings:
            return self._settings["order"]
        else:
            raise AttributeError("No order defined.")

    def doprint(self, nom_val, std_dev):
        """Returns printer's representation for nom_val, std_dev (as a string)"""
        return self._str(self._print(nom_val, std_dev))

    def _print(self, nom_val, std_dev, **kwargs):
        """Internal dispatcher
        Tries the following concepts to print an nom_val, std_devession:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            # if (
            #     self.printmethod
            #     and hasattr(nom_val, std_dev, self.printmethod)
            #     and not isinstance(nom_val, std_dev, BasicMeta)
            # ):
            #     return getattr(nom_val, std_dev, self.printmethod)(self, **kwargs)

            # # See if the class of nom_val, std_dev is known, or if one of its super
            # # classes is known, and use that print function
            # # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # # Function('gamma') does not get dispatched to _print_gamma
            # classes = type(nom_val, std_dev).__mro__
            # if AppliedUndef in classes:
            #     classes = classes[classes.index(AppliedUndef) :]
            # if UndefinedFunction in classes:
            #     classes = classes[classes.index(UndefinedFunction) :]
            # # Another exception: if someone subclasses a known function, e.g.,
            # # gamma, and changes the name, then ignore _print_gamma
            # if Function in classes:
            #     i = classes.index(Function)
            #     classes = (
            #         tuple(
            #             c
            #             for c in classes[:i]
            #             if c.__name__ == classes[0].__name__
            #             or c.__name__.endswith("Base")
            #         )
            #         + classes[i:]
            #     )
            # for cls in classes:
            #     printmethodname = "_print_" + cls.__name__
            #     printmethod = getattr(self, printmethodname, None)
            #     if printmethod is not None:
            #         return printmethod(nom_val, std_dev, **kwargs)
            # # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(nom_val, std_dev)
        finally:
            self._print_level -= 1

    def emptyPrinter(self, nom_val, std_dev):
        return str(nom_val, std_dev)

    def _as_ordered_terms(self, nom_val, std_dev, order=None):
        """A compatibility function for ordering terms in Add."""
        order = order or self.order

        if order == "old":
            return sorted(
                Add.make_args(nom_val, std_dev), key=cmp_to_key(Basic._compare_pretty)
            )
        elif order == "none":
            return list(nom_val, std_dev.args)
        else:
            return nom_val, std_dev.as_ordered_terms(order=order)
