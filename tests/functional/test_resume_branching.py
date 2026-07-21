"""Resuming a COPIED checkpoint must write beside the copy, not its birthplace.

This is what makes a two-phase experiment possible: burn one population in to
equilibrium, copy that output directory per experimental arm, and resume each copy
under a different regime. Every arm then shares an identical equilibrated ancestor.

init_resume() used to hand recordingmanager the config path stored INSIDE the
checkpoint, which is the ancestor's. Every arm therefore wrote its output back into the
ancestor's directory: arms silently overwrote each other and the ancestor, and running
them concurrently raced over the same checkpoint file (observed on the cluster as
`FileNotFoundError: .../checkpoint -> .../checkpoint.bak` when 12 arms started at once).
"""

import subprocess
import shutil

import pytest
import yaml


def _config(steps):
    return dict(
        RANDOM_SEED=1, STEPS_PER_SIMULATION=steps, AGE_LIMIT=20, MATURATION_AGE=4,
        INITIAL_POPULATION_SIZE=200, RESOURCE_MAXIMUM_AMOUNT=200,
        RESOURCE_ADDITIVE_GROWTH=200, RESOURCE_INITIAL_AMOUNT=200,
        REPRODUCTION_MODE="sexual", MAX_OFFSPRING_NUMBER=3,
        G_muta_initpheno=1e-3, G_surv_initgeno=0.9, G_repr_initgeno=0.9,
        CHECKPOINT_RATE=100, SNAPSHOT_RATE=200, PICKLE_RATE=200, TE_RATE=200,
        LOGGING_RATE=200, POPGENSTATS_RATE=0,
    )


def _run(*args):
    return subprocess.run(["aegis", "sim", *args], capture_output=True, text=True, timeout=300)


def _rows(path):
    return sum(1 for _ in open(path))


@pytest.fixture(scope="module")
def ancestor(tmp_path_factory):
    d = tmp_path_factory.mktemp("branch")
    cfg = d / "anc.yml"
    yaml.safe_dump(_config(200), open(cfg, "w"), sort_keys=True)
    r = _run("-c", str(cfg))
    assert r.returncode == 0, r.stderr[-2000:]
    return d, cfg


class TestResumeBranching:

    def test_arms_do_not_write_into_the_ancestor(self, ancestor):
        d, cfg = ancestor
        anc_series = d / "anc" / "popsize_before_reproduction.csv"
        before = _rows(anc_series)

        for arm, extend in (("arm_a", 300), ("arm_b", 400)):
            shutil.copytree(d / "anc", d / arm)
            shutil.copy(cfg, d / f"{arm}.yml")
            r = _run("-c", str(d / f"{arm}.yml"), "-r", "--extend", str(extend))
            assert r.returncode == 0, r.stderr[-2000:]

        assert _rows(anc_series) == before, "the ancestor was written to by an arm"
        assert _rows(d / "arm_a" / "popsize_before_reproduction.csv") == 300
        assert _rows(d / "arm_b" / "popsize_before_reproduction.csv") == 400

    def test_arms_can_diverge_by_override(self, ancestor):
        """Two arms off one ancestor, differing only in an overridden parameter."""
        d, cfg = ancestor
        for arm, growth in (("arm_lo", 50), ("arm_hi", 800)):
            shutil.copytree(d / "anc", d / arm)
            shutil.copy(cfg, d / f"{arm}.yml")
            r = _run("-c", str(d / f"{arm}.yml"), "-r", "--extend", "320",
                     "--override", f"RESOURCE_ADDITIVE_GROWTH={growth}")
            assert r.returncode == 0, r.stderr[-2000:]

        lo = yaml.safe_load(open(d / "arm_lo" / "final_config.yml"))
        hi = yaml.safe_load(open(d / "arm_hi" / "final_config.yml"))
        assert lo["RESOURCE_ADDITIVE_GROWTH"] == 50
        assert hi["RESOURCE_ADDITIVE_GROWTH"] == 800

    def test_resume_in_place_still_works(self, ancestor):
        """The ordinary case -- resuming a run where it was created -- is unaffected."""
        d, cfg = ancestor
        inplace_cfg = d / "inplace.yml"
        yaml.safe_dump(_config(150), open(inplace_cfg, "w"), sort_keys=True)
        assert _run("-c", str(inplace_cfg)).returncode == 0
        r = _run("-c", str(inplace_cfg), "-r", "--extend", "250")
        assert r.returncode == 0, r.stderr[-2000:]
        assert _rows(d / "inplace" / "popsize_before_reproduction.csv") == 250
