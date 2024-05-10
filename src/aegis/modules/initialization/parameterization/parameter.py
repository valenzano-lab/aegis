import logging


class Parameter:
    def __init__(
        self,
        key,
        name,
        domain,
        default,
        info,
        dtype,
        drange,
        inrange=lambda x: True,
        serverrange=lambda x: True,
        serverrange_info="",
        evalrange=None,
        presets={},
    ):
        self.key = key
        self.name = name
        self.domain = domain
        self.default = default
        self.info = info
        self.dtype = dtype
        self.drange = drange
        self.inrange = inrange
        self.serverrange = serverrange
        self.serverrange_info = serverrange_info
        self.evalrange = evalrange
        self.presets = presets

    def convert(self, value):
        if value is None or value == "":
            return self.default
        elif self.dtype == bool:
            return value in ("True", "true", True)
        else:
            return self.dtype(value)

    def valid(self, value):
        # Not valid if wrong data type
        if not isinstance(value, self.dtype):
            logging.error(f"Value {value} is not of valid type {self.dtype} but of type {type(value)}")
            return False

        # Not valid if not in range
        if not self.inrange(value):
            return False

        # Valid
        return True

    def get_name(self):
        if self.name:
            return self.name
        name = self.key.replace("_", " ").strip().lower()
        return name

    def validate_dtype(self, value):
        can_be_none = self.default is None
        # given custom value is None which is a valid data type
        if can_be_none and value is None:
            return
        # given custom value is of valid data type
        if isinstance(value, self.dtype):
            return
        # given custom value is int but float is valid
        if self.dtype is float and isinstance(value, int):
            return
        raise TypeError(
            f"You set {self.key} to be '{value}' which is of type {type(value)} but it should be {self.dtype} {'or None' if can_be_none else ''}"
        )

    def validate_inrange(self, value):
        if self.inrange(value):
            return
        raise ValueError(f"{self.key} is set to be '{value}' which is outside of the valid range '{self.drange}'.")

    def validate_serverrange(self, value):
        if self.serverrange(value):
            return
        raise ValueError(
            f"{self.key} is set to be '{value}' which is outside of the valid server range '{self.drange}'."
        )

    def generate_full_evalrange(self):
        if self.evalrange is not None:
            n_datapoints = 10
            from numpy import linspace

            return linspace(self.evalrange[0], self.evalrange[1], n_datapoints)
