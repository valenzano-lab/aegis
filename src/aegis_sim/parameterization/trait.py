from aegis_sim import constants


class Trait:
    """Genetic trait

    Contains data on traits encoded in the genome.
    """

    def __init__(self, name, cnf):
        def get(key):
            return getattr(cnf, f"G_{name}_{key}")

        self.name = name

        # Attributes set by the configuration files
        self.evolvable = get("evolvable")
        self.agespecific = get("agespecific")
        self.interpreter = get("interpreter")
        self.lo = get("lo")
        self.hi = get("hi")
        self.initgeno = get("initgeno")
        self.initpheno = get("initpheno")

        # Determine the number of loci encoding the trait
        if self.evolvable:
            if self.agespecific is True:  # one locus per age
                self.length = cnf.AGE_LIMIT
            elif self.agespecific is False:  # one locus for all ages
                self.length = 1
            else:  # custom number of loci
                self.length = self.agespecific
        else:  # no loci for a constant trait
            self.length = 0

        self._validate()

        # Infer positions in the genome
        # self.start = start
        # self.end = self.start + self.length
        # self.slice = slice(self.start, self.end)

        self.start = cnf.AGE_LIMIT * constants.starting_site(self.name)
        self.end = self.start + cnf.AGE_LIMIT
        self.slice = slice(self.start, self.end)

    def _validate(self):
        """Check whether input parameters are legal."""
        if not isinstance(self.evolvable, bool):
            raise TypeError

        if not 0 <= self.initgeno <= 1:
            raise ValueError

        if self.evolvable:
            # if not isinstance(self.agespecific, bool):
            #     raise TypeError

            if self.interpreter not in (
                "uniform",
                "exp",
                "binary",
                "binary_exp",
                "binary_switch",
                "switch",
                "linear",
                "single_bit",
                "const1",
                "threshold",
            ):
                raise ValueError(f"{self.interpreter} is not a valid interpreter type")

            if not 0 <= self.lo <= 1:
                raise ValueError

            if not 0 <= self.hi <= 1:
                raise ValueError

    def __len__(self):
        """Return number of loci used to encode the trait."""
        return self.length

    def __str__(self):
        return self.name
