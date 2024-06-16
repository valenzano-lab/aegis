import functools
import numpy as np

from aegis.hermes import hermes


def stan_age(age):
    return age / hermes.parameters.AGE_LIMIT


class GPM_decoder:
    """
    Example input...
    PHENOMAP:
        "AP1, 7":
            - - "surv"
            - "agespec"
            - - 0.1
            - - "repr"
            - "agespec"
            - - 0.1
        "MA1, 13":
            - - "surv"
            - "agespec"
            - - 0.1
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
                age = hermes.rng.integers(0, hermes.parameters.AGE_LIMIT)
                magnitude = func(age)
                add_to_phenolist(i, encoding["trait"], age, magnitude)
            else:
                for age in range(hermes.parameters.AGE_LIMIT):
                    magnitude = func(age)
                    add_to_phenolist(i, encoding["trait"], age, magnitude)

    @staticmethod
    def __resolve_function(func, params):

        scale = params

        # def const(intercept, age):
        #     return intercept

        # def linear(intercept, slope, age):
        #     return intercept + slope * age

        # def gompertz(baseline, aging_rate, age):
        #     """
        #     Exponential increase in mortality.
        #     When aging_rate = 0, there is no aging.
        #     Baseline is intercept.
        #     """
        #     return [-1, 1][aging_rate > 0] * baseline * np.exp(abs(aging_rate) * age)

        # def makeham(intercept, baseline, aging_rate, age):
        #     return intercept + gompertz(baseline=baseline, aging_rate=aging_rate, age=age)

        # def siler(baseline_infancy, slope_infancy, intercept, baseline, aging_rate, age):
        #     return baseline_infancy * np.exp(-slope_infancy * age) + makeham(
        #         intercept, baseline=baseline, aging_rate=aging_rate, age=age
        #     )

        # def _normal(pair):
        #     mean, std = pair
        #     return hermes.rng.normal(mean, std)

        # def _exponential(param):
        #     if param > 0:
        #         return hermes.rng.exponential(param)
        #     else:
        #         return -hermes.rng.exponential(-param)

        def _beta(a=3, b=3):
            return np.random.beta(a, b) * 2 - 1
            # return np.random.beta(a, b)

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

        if func == "flat" or func == "agespec":
            return functools.partial(acc, _beta(), 0, 0)
        elif func == "lin":
            return functools.partial(acc, _beta() / 2, _beta(), 0)
        elif func == "acc":
            return functools.partial(acc, _beta(), 0, _beta())
        elif func == "exp":
            return functools.partial(exp, _beta())
        elif func == "hump":
            return functools.partial(hump, _beta(), _beta(1, 1), _beta())
        elif func == "sigm":
            return functools.partial(sigm, _beta(), _beta(1, 1))
        else:
            raise Exception(f"{func} not allowed.")

        # if func == "const" or func == "agespec":
        #     return functools.partial(const, _exponential(params[0]))
        # elif func == "linear":
        #     assert len(params) == 2, "Two parameters are needed for a linear block."
        #     return functools.partial(linear, _exponential(params[0]), _exponential(params[1]))
        # elif func == "gompertz":
        #     assert len(params) == 2, "Two parameters are needed for a Gompertz block."
        #     return functools.partial(gompertz, _exponential(params[0]), _exponential(params[1]))
        # else:
        #     raise Exception(f"{func} not allowed.")

    @staticmethod
    def __resolve_function_old(func, params):

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
            return [-1, 1][aging_rate > 0] * baseline * np.exp(abs(aging_rate) * age)

        def makeham(intercept, baseline, aging_rate, age):
            return intercept + gompertz(baseline=baseline, aging_rate=aging_rate, age=age)

        def siler(baseline_infancy, slope_infancy, intercept, baseline, aging_rate, age):
            return baseline_infancy * np.exp(-slope_infancy * age) + makeham(
                intercept, baseline=baseline, aging_rate=aging_rate, age=age
            )

        def _normal(pair):
            mean, std = pair
            return hermes.rng.normal(mean, std)

        def _exponential(param):
            if param > 0:
                return hermes.rng.exponential(param)
            else:
                return -hermes.rng.exponential(-param)

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
