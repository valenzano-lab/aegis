"""Fit the selection coefficient s from an AEGIS selection.csv log.

Reads <sim_dir>/selection/selection.csv and fits the linear regression
    log( p / (1 - p) )  ~  a + s * t
over the post-injection portion of the trajectory. Returns s as the slope.

Usage:
    python runs/fit_s.py <sim_dir>
        # writes <sim_dir>/selection/fit_s.png and prints s with 95% CI

Notes:
    - Frequencies of 0 or 1 are clipped to (eps, 1-eps) to keep the log finite.
    - The fit assumes additive selection (constant per-generation selection
      pressure on the focal allele) — a reasonable first approximation for
      small s in age-structured populations. For age-specific traits the
      effective s also depends on which age the locus is expressed at.
    - The "injection step" is read from <sim_dir>/final_config.yml or, if that
      isn't accessible, from the first step where the carrier count visibly
      jumps. Falls back to fitting the whole trajectory if injection step
      can't be inferred.
"""

import pathlib
import sys

import numpy as np
import pandas as pd


def find_injection_step(sim_dir: pathlib.Path) -> int:
    cfg_path = sim_dir / "final_config.yml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            return int(cfg.get("ALLELE_INJECTION_STEP", 0))
        except Exception:
            pass
    return 0


def main(sim_dir: pathlib.Path) -> int:
    sel_path = sim_dir / "selection" / "selection.csv"
    if not sel_path.exists():
        print(f"No selection.csv at {sel_path}; set ALLELE_INJECTION_STEP > 0 in the config.")
        return 1
    df = pd.read_csv(sel_path)
    if len(df) < 5:
        print(f"selection.csv has only {len(df)} rows; need more to fit.")
        return 1

    injection_step = find_injection_step(sim_dir)
    post = df[df["step"] >= injection_step].copy() if injection_step > 0 else df.copy()
    if len(post) < 5:
        print(f"Only {len(post)} rows post-injection; need >= 5.")
        return 1

    eps = 1.0 / (4 * post["n_alleles"].max())
    p = np.clip(post["allele_freq"].to_numpy(), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    t = post["step"].to_numpy().astype(float)
    t = t - t[0]  # zero-base time at the injection step for interpretability

    # Linear fit: logit ~ a + s * t
    n = len(t)
    coeffs, cov = np.polyfit(t, logit, deg=1, cov=True)
    s_hat, intercept = float(coeffs[0]), float(coeffs[1])
    s_se = float(np.sqrt(cov[0, 0]))
    s_ci = (s_hat - 1.96 * s_se, s_hat + 1.96 * s_se)

    print(f"  rows fit: {n}")
    print(f"  injection step: {injection_step}")
    print(f"  s_hat = {s_hat:.5f}")
    print(f"  95% CI = ({s_ci[0]:.5f}, {s_ci[1]:.5f})")
    print(f"  intercept = {intercept:.4f}  (logit at t=0)")
    print(f"  starting freq (fitted) = {1 / (1 + np.exp(-intercept)):.4f}")
    print(f"  pre-injection rows ignored: {len(df) - n}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_p, ax_logit) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_p.plot(df["step"], df["allele_freq"], marker=".", linestyle="-", ms=3, label="observed")
    ax_p.axvline(injection_step, color="red", linestyle="--", alpha=0.5, label=f"injection (step {injection_step})")
    ax_p.set_ylabel("allele frequency  p")
    ax_p.set_ylim(0, 1)
    ax_p.legend(loc="best")

    ax_logit.plot(post["step"], logit, marker=".", linestyle="", ms=3, label="post-injection logit(p)")
    fit_line = intercept + s_hat * t
    ax_logit.plot(post["step"], fit_line, color="black", linewidth=1.5, label=f"fit  s = {s_hat:.4f}")
    ax_logit.set_xlabel("step")
    ax_logit.set_ylabel("log(p / (1 - p))")
    ax_logit.legend(loc="best")

    fig.suptitle(f"Selection coefficient fit — s = {s_hat:.4f}  [95% CI {s_ci[0]:.4f}, {s_ci[1]:.4f}]")
    fig.tight_layout()
    out = sim_dir / "selection" / "fit_s.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runs/fit_s.py <sim_dir>")
        sys.exit(2)
    sys.exit(main(pathlib.Path(sys.argv[1])))
