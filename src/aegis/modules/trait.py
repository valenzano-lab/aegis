class Trait:
    """Genetic trait

    Contains data on traits encoded in the genome.
    """

    legal = ("surv", "repr", "neut", "muta")

    def __init__(self, name, params, start):
        def get(key):
            return params[f"G_{name}_{key}"]

        self.name = name

        # Attributes set by the configuration files
        self.evolvable = get("evolvable")
        self.agespecific = get("agespecific")
        self.interpreter = get("interpreter")
        self.lo = get("lo")
        self.hi = get("hi")
        self.initial = get("initial")

        # Determine the number of loci encoding the trait
        if self.evolvable:
            if self.agespecific:  # one locus per age
                self.length = params["MAX_LIFESPAN"]
            else:  # one locus for all ages
                self.length = 1
        else:  # no loci for a constant trait
            self.length = 0

        self._validate()

        # Infer positions in the genome
        self.start = start
        self.end = self.start + self.length
        self.slice = slice(self.start, self.end)

    def _validate(self):
        """Check whether input parameters are legal."""
        if not isinstance(self.evolvable, bool):
            raise TypeError

        if not 0 <= self.initial <= 1:
            raise ValueError

        if self.evolvable:
            if not isinstance(self.agespecific, bool):
                raise TypeError

            if self.interpreter not in (
                "uniform",
                "exp",
                "binary",
                "binary_exp",
                "binary_switch",
                "switch",
            ):
                raise ValueError

            if not 0 <= self.lo <= 1:
                raise ValueError

            if not 0 <= self.hi <= 1:
                raise ValueError

    def __len__(self):
        """Return number of loci used to encode the trait."""
        return self.length

    def __str__(self):
        return self.name
