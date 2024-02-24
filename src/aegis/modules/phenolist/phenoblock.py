import numpy as np
import functools

from aegis import var
from aegis import cnf


class Phenoblock:
    def __init__(self, encoding):
        self.encoding = encoding

        self.name = encoding[0]
        self.n = int(encoding[1])

        self.trait = encoding[2]

        # timing
        self.agespecific = encoding[3]  # True if only one age affected, False if whole interval affected
        self.domain = "life"  # 10, 10-40, life, mature, immature

        # magnitude
        self.effectfunc = encoding[5]  # const, linear, sigmoidal, exponential
        self.effectparams = encoding[6]  # ((0.01, 0.01)) or ((0.01, 0.01), (0.02, 0.007))

        # init
        self.sites = []

    @staticmethod
    def decode(script):
        """
        block_name, block_size, trait, domain, function, parameters
        """
        return

    @staticmethod
    def resolve(n, trait, _domain, m_func_, m_parameters_, n0=0):

        domain = Phenoblock.resolve_domain(_domain)

        for site in range(n0, n0 + n):

            magnitude_func = Phenoblock.resolve_function(m_func_, m_parameters_)
            for age in domain:
                magnitude = magnitude_func(age)

                yield site, trait, age, magnitude

    @staticmethod
    def resolve_domain(domain):
        """
        Possible domains:
        - life
        - mature
        - menopausal
        - range (inclusive)
        - age
        """

        domain = str(domain)  # in case it is only age, given as an int; last case

        if domain == "life":
            return list(range(cnf.MAX_LIFESPAN))
        elif domain == "immature":
            return list(range(cnf.MATURATION_AGE))
        elif domain == "mature":
            return list(range(cnf.MATURATION_AGE, cnf.MAX_LIFESPAN))
        elif domain == "menopausal":
            return list(range(cnf.MENOPAUSE, cnf.MAX_LIFESPAN))
        elif "-" in domain:
            # inclusive
            from_, to_ = (int(x) for x in domain.split("-"))
            assert from_ >= 0
            assert to_ < cnf.MAX_LIFESPAN
            return list(range(from_, to_ + 1))
        else:
            age = int(domain)
            assert 0 <= age < cnf.MAX_LIFESPAN
            return [age]

    @staticmethod
    def resolve_function(func, params):

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

        if func == "const":
            return functools.partial(const, _normal(params[0]))
        elif func == "linear":
            return functools.partial(linear, _normal(params[0]), _normal(params[1]))
        elif func == "gompertz":
            return functools.partial(gompertz, _normal(params[0]), _normal(params[1]))
        else:
            raise Exception(f"{func} not allowed.")

    # @staticmethod
    # def resolve_agespecific(n, trait, _domain, mean, std):
    #     # (site, trait, age, magnitude)

    #     domain = Phenoblock.resolve_domain(_domain)

    #     for site in range(n):
    #         age = rng.choice(domain)
    #         magnitude = rng.normal(mean, std)

    #         yield site, trait, age, magnitude
