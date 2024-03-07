import functools
import numpy as np

from aegis.pan import var


class Genst:
    def __init__(self, encodings):
        self.blocks = {}
        self.n = 0

        for (name, n), quartets in encodings.items():
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
    def __init__(self, name, n, position, MAX_LIFESPAN=20):
        self.name = name
        self.n = n

        self.position = position

        self.encodings = []

        self.phenolist = []

        self.MAX_LIFESPAN = MAX_LIFESPAN

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
                age = var.rng.integers(0, self.MAX_LIFESPAN)
                magnitude = func(age)
                add_to_phenolist(i, encoding["trait"], age, magnitude)
            else:
                for age in range(self.MAX_LIFESPAN):
                    magnitude = func(age)
                    add_to_phenolist(i, encoding["trait"], age, magnitude)

    @staticmethod
    def __resolve_function(func, params):

        def const(intercept, age):
            return intercept

        def linear(intercept, slope, age):
            return intercept + slope * age

        def gompertz(baseline, aging_rate, age):
            """
            Exponential increase in mortality.
            When aging_rate = 0, there is no aging.
            Baseline is intercept.
            """
            return baseline * np.exp(aging_rate * age)

        def makeham(intercept, baseline, aging_rate, age):
            return intercept + gompertz(baseline=baseline, aging_rate=aging_rate, age=age)

        def siler(baseline_infancy, slope_infancy, intercept, baseline, aging_rate, age):
            return baseline_infancy * np.exp(-slope_infancy * age) + makeham(
                intercept, baseline=baseline, aging_rate=aging_rate, age=age
            )

        def _normal(pair):
            mean, std = pair
            return var.rng.normal(mean, std)

        def _exponential(param):
            return var.rng.exponential(param)

        if func == "const" or func == "agespec":
            return functools.partial(const, _exponential(params[0]))
        elif func == "linear":
            assert len(params) == 2, "Two parameters are needed for a linear block."
            return functools.partial(linear, _exponential(params[0]), _exponential(params[1]))
        elif func == "gompertz":
            assert len(params) == 2, "Two parameters are needed for a Gompertz block."
            return functools.partial(gompertz, _exponential(params[0]), _exponential(params[1]))
        else:
            raise Exception(f"{func} not allowed.")
