import numpy as np
import pytest

from aegis_sim.parameterization.trait import Trait
from aegis_sim.submodels.genetics.composite.interpreter import Interpreter


def test_custom_weighted_interpreter_normalizes_weights():
    interpreter = Interpreter(BITS_PER_LOCUS=4, THRESHOLD=None)
    loci = np.array([[[True, False, True, False]]])

    values = interpreter.call(
        loci,
        interpreter_kind="custom_weighted",
        custom_weights=[10, 3, 1, 1],
    )

    assert values[0, 0] == pytest.approx(11 / 15)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1, 1, 1], "exactly 4 weights"),
        ([1, -1, 1, 1], "cannot contain negative"),
        ([0, 0, 0, 0], "positive total"),
    ],
)
def test_custom_weight_validation(weights, message):
    class Config:
        AGE_LIMIT = 10
        BITS_PER_LOCUS = 4
        G_surv_evolvable = True
        G_surv_agespecific = False
        G_surv_interpreter = "custom_weighted"
        G_surv_initgeno = 0.5
        G_surv_custom_weights = weights
        G_surv_initpheno = 0.5
        G_surv_lo = 0
        G_surv_hi = 1

    with pytest.raises(ValueError, match=message):
        Trait("surv", Config, "composite", MODIF_GENOME_SIZE=1, start_position=0)
