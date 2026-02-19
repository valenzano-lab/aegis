import numpy as np


def init(self, custom_config_path, pickle_path, RANDOM_SEED):
    self.steps = 1
    self.custom_config_path = custom_config_path
    self.pickle_path = pickle_path
    self.random_seed = np.random.randint(1, 10**6) if RANDOM_SEED is None else RANDOM_SEED
    # TODO: Consolidate RNG usage across the codebase. Currently both the legacy
    # global RNG (np.random.*) and the new-style Generator (variables.rng) are used
    # in different modules. Both are seeded here so simulations are reproducible, but
    # the split makes it easy to accidentally shift the random stream during refactors.
    # Affected modules using legacy np.random: population.initialize, matingmanager,
    # abiotic, envdrift, recombination, gpm_decoder, popgenstats.
    np.random.seed(self.random_seed)
    self.rng = np.random.default_rng(self.random_seed)


def restore_from_checkpoint(self, checkpoint):
    """Restore variables state from a Checkpoint object."""
    self.steps = checkpoint.step
    self.custom_config_path = checkpoint.custom_config_path
    self.pickle_path = None
    self.random_seed = checkpoint.random_seed
    # Restore both RNG states
    np.random.set_state(checkpoint.legacy_rng_state)
    self.rng = np.random.default_rng()
    self.rng.bit_generator.state = checkpoint.rng_state
