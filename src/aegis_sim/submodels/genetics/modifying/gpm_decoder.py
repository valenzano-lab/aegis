import functools
import numpy as np


from aegis_sim.parameterization import parametermanager


def stan_age(age):
    return age / parametermanager.parameters.AGE_LIMIT


class GPM_decoder:
    """Converter of simple user input into a genotype-phenotype map (as a list or a matrix).

    --- Example input ---

    PHENOMAP:
        "AP1, 7": # name of block, number of sites
            - ["surv", "agespec", 0.1] # trait, age-dependency, scale
            - ["repr", "agespec", -0.1]
        "MA1, 13":
            - ["surv", "agespec", -0.1]

    # Make sure you set:
    GENARCH_TYPE: modifying
    BITS_PER_LOCUS: 1

    # Make sure your genome size is big enough for all sites set up
    MODIF_GENOME_SIZE: 2000

    ---
    """

    def __init__(self, config_PHENOMAP):
        self.blocks = {}
        self.n = 0

        for namen, quartets in config_PHENOMAP.items():
            # YML dict cannot take tuple as key, so it is fused as f'{name}, {n}'
            name, n = namen.split(",")
            n = int(n)
            self.__add_genblock(name, n)
            for trait, agefunc, funcparam in quartets:
                self.__add_encoding(name, trait, agefunc, funcparam)

    def get_total_phenolist(self):
        phenolist = []
        for block in self.blocks.values():
            phenolist.extend(block.get_phenolist())
        return phenolist

    def __len__(self):
        return self.n

    # PRIVATE
    def __add_genblock(self, name, n):
        assert name not in self.blocks.keys(), f"Block with name {name} already exists."

        genblock = Genblock(name=name, n=n, position=self.n)
        self.blocks[name] = genblock
        self.n += genblock.n

    def __add_encoding(self, name, trait, agefunc, funcparam):
        self.blocks[name].add_encoding(trait, agefunc, funcparam)


class Genblock:
    def __init__(self, name, n, position):
        self.name = name
        self.n = n

        self.position = position

        self.encodings = []

        self.phenolist = []

    def add_encoding(self, trait, agefunc, funcparam):
        encoding = {
            "trait": trait,
            "agefunc": agefunc,
            "funcparam": funcparam,
        }
        self.encodings.append(encoding)
        self.__decode(encoding=encoding)

    def get_phenolist(self):
        for index, trait, age, magnitude in self.phenolist:
            yield index + self.position, trait, age, magnitude

    def __decode(self, encoding):

        def add_to_phenolist(index, trait, age, magnitude):
            self.phenolist.append([index, trait, age, magnitude])

        for i in range(self.n):
            func = Genblock.__resolve_function(encoding["agefunc"], encoding["funcparam"])
            if encoding["agefunc"] == "agespec":
                age = np.random.randint(0, parametermanager.parameters.AGE_LIMIT)
                magnitude = func(age)
                add_to_phenolist(i, encoding["trait"], age, magnitude)
            else:
                for age in range(parametermanager.parameters.AGE_LIMIT):
                    magnitude = func(age)
                    add_to_phenolist(i, encoding["trait"], age, magnitude)

    @staticmethod
    def __resolve_function(func, params):

        scale = params

        # PARAMETER DISTRIBUTIONS
        def _beta_symmetrical(a=3, b=3):
            """
            a=3, b=3 -> bell-shape
            a=1, b=3 -> decay (highest around 0, decreases with distance from 0)
            a=1, b=1 -> flat
            """
            return np.random.beta(a, b) * 2 - 1

        def _beta_onesided(a, b):
            return np.random.beta(a, b)

        # PHENOTYPE(AGE) functions
        # These functions are used to compute phenotypic effects for every age
        # They are going to be partial-ed with beta-shaped parameters (above)
        # And then they take in age as a variable

        def acc(intercept, slope, acc, age):
            x = stan_age(age)
            y = intercept + slope * x + acc * x**3
            return scale * y

        def exp(exp, age):
            x = stan_age(age)
            sign = [-1, 1][exp > 0]
            y = sign * ((x + 1) ** (abs(exp) * 5) - 1)
            return scale * y

        def hump(amplitude, x0, width, age):
            x = stan_age(age)
            midpoint = 0.5
            y = amplitude * np.exp(-((x - midpoint - x0) ** 2) / (0.02 + (width * 0.2) ** 2))
            return scale * y

        def sigm(slope, shift, age):
            x = stan_age(age)
            k = slope * 50
            midpoint = 0.5
            y = 1 / (1 + np.exp((-1) * k * (x - midpoint - shift / 2)))
            return scale * y - scale / 2

        # Assembling it together
        if func == "agespec":
            return functools.partial(acc, _beta_onesided(a=1, b=3), 0, 0)
        elif func == "lin":
            # linearly changing with age
            return functools.partial(acc, _beta_symmetrical() / 2, _beta_symmetrical(), 0)
        elif func == "acc":
            # exponentially changing with age
            return functools.partial(acc, _beta_symmetrical(), 0, _beta_symmetrical())
        elif func == "exp":
            # exponentially changing with age
            return functools.partial(exp, _beta_symmetrical())
        elif func == "hump":
            # not age-specific but with an age-hump
            return functools.partial(hump, _beta_symmetrical(), _beta_symmetrical(1, 1), _beta_symmetrical())
        elif func == "sigm":
            # sigmoidal across age
            return functools.partial(sigm, _beta_symmetrical(), _beta_symmetrical(1, 1))
        else:
            raise Exception(f"{func} not allowed.")
