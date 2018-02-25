########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Config                                                      #
# Description: Simulation configuration information imported from a    #
#   config file and saved as a dictionary object for use by other      #
#   classes. The dictionary keys are linked to both values and         #
#   information entries for identifying the function of each entry.    #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
import copy, imp, math, pickle
from .functions import deep_key, deep_eq

class Config(dict):
    """Object derived from imported config module."""

    def __init__(self, filepath):
        """Import basic attributes from a config file and set data
        descriptors."""
        self.__valdict__ = {}
        self.__infdict__ = {}
        c = imp.load_source('ConfFile', filepath)
        for k in [d for d in dir(c) if not d.startswith("_")]:
            self[k] = getattr(c, k) # Copy all keys from config file
        self.check()

    # Check for invalid config construction
    def check(self):
        """Confirm that population parameters are compatible before
        starting simulation."""
        states = ['asexual','recombine_only','assort_only','sexual']
        if self["repr_mode"] not in states:
            s0 = "Invalid reproductive mode: '{}'.".format(self["repr_mode"])
            s1 = " Must be one of: {}.".format(", ".join(states))
            print self["repr_mode"]
            raise ValueError(s0 + s1)
        if self["maturity"] > self["max_ls"] - 2:
            s = "Age of maturity must be at least 2 less than max lifespan."
            raise ValueError(s)
        if self["repr_offset"] < self["max_ls"]:
            s = "Offset for reproductive loci must be >= max lifespan."
            raise ValueError(s)
        if self["neut_offset"] - self["repr_offset"] < self["max_ls"]:
            s = "Difference between offset values must be >= max lifespan."
            raise ValueError(s)
        return True

    def make_params(self):
        key_list = ["repr_mode", "chr_len", "n_base", "maturity", "start_pop",
                "max_ls", "g_dist", "repr_offset", "neut_offset", "object_max_age",\
                "prng"]
        return dict([(k, self[k]) for k in key_list])

    # Generate derived attributes

    def generate(self):
        """Generate derived configuration attributes from simple ones and
        add to configuration object."""
        self.check()
        # Compute random seed
        def get_random_seed(path):
            try:
                infile = open(path, "rb")
                obj = pickle.load(infile)
                if not isinstance(obj, tuple):
                    s = "Random seed must be a tuple representing the internal"\
                        " state of the generator"
                    raise ImportError(s)
                infile.close()
                return obj
            except IOError:
                print "Random seed import failed: no such file under {}".format(path)
        self["prng"] = np.random.RandomState()
        if isinstance(self["random_seed"], str) and self["random_seed"]!="":
            self["prng"].set_state(get_random_seed(self["random_seed"]))
        else: self["random_seed"] = self["prng"].get_state()
        # Genome structure
        self["genmap"] = np.concatenate([
            np.arange(self["max_ls"]),
            np.arange(self["maturity"],self["max_ls"]) + self["repr_offset"],
            np.arange(self["n_neutral"]) + self["neut_offset"]], 0)
        self["genmap_argsort"] = np.argsort(self["genmap"])
        self["chr_len"] = len(self["genmap"]) * self["n_base"]
        self["n_states"] = 2*self["n_base"]+1
        # Survival and reproduction
        self["death_bound"] = np.array(self["death_bound"])
        self["repr_bound"] = np.array(self["repr_bound"])
        if self["repr_mode"] in ["sexual", "assort_only"]:
        # Double fertility in sexual case to control for relative
        # contribution to offspring
            self["repr_bound"] *= 2
        self["surv_bound"] = 1-self["death_bound"][::-1]
        self["surv_step"] = np.diff(self["surv_bound"])/(self["n_states"]-1)
        self["repr_step"] = np.diff(self["repr_bound"])/(self["n_states"]-1)
        self["s_range"] = np.linspace(self["surv_bound"][0],
            self["surv_bound"][1], self["n_states"])
        self["r_range"] = np.linspace(self["repr_bound"][0],
            self["repr_bound"][1], self["n_states"])
        # Check whether stage counting is automatic
        self["auto"] = (self["n_stages"] == "auto")
        self["object_max_age"] = self["n_stages"] if not self["auto"]\
                else self["max_stages"]
        # Params dict
        self["params"] = self.make_params()
        # Compute automatic stage numbering (if required)
        self.autostage()

    def autostage(self):
        """Compute automatic running behaviour ... UNTESTED"""
        if self["auto"]:
            # Compute analytical parameters
            alpha, beta = self["m_rate"], self["m_rate"]*self["m_ratio"]
            y = self["g_dist"]["n"]
            x = 1-y
            k = math.log10(self["delta"]*(alpha+beta)/abs(alpha*y-beta*x)) / \
                    math.log10(abs(1-alpha-beta))
            # Assign generation threshold
            self["min_gen"] = int(k*self["scale"])

        # Compute snapshot generations
        ss_key = "generations" if self["auto"] else "stages"
        ss_max = self["min_gen"] if self["auto"] else self["n_stages"] - 1
        self["snapshot_{}".format(ss_key)] = np.around(np.linspace(
                0, ss_max, self["n_snapshots"])).astype(int)
        if self["auto"]:
            self["snapshot_generations_remaining"] = np.copy(
                    self["snapshot_generations"])

    # COMPARISON

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return deep_eq(self, other, True)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    # COPYING

    def copy(self):
        self_copy = copy.deepcopy(self)
        self_copy["prng"] = self["prng"]
        self_copy["params"]["prng"] = self["params"]["prng"]
        return self_copy
