"""Two-phase experiment: equilibrium life history, then the released dynamics.

A single run holds both phases -- burn-in under constant, regulated resources, then a
resumed extension with regulation released and the resource dynamics switched on. The
neutral locus confirms the burn-in reached equilibrium before the switch (see
check_equilibration.py), so the oscillation is the response of an EQUILIBRATED
population, not a transient from initialization.

  Left  survival px(age): NAVY at the release (the aging evolved under constant,
        regulated resources) and ORANGE at the end of the released phase, so the
        shift in the schedule under resource stress is visible. Dashed lines are the
        corresponding survivorship lx = prod px. Started flat (non-aging); the gap
        from the 0.95 line is evolved senescence.
  Right N(t) and R(t) across the phase boundary -- flat and pinned while regulated,
        then oscillating once released. The vertical line marks the release step.

Usage:
    python runs/plot_burnin_and_release.py ~/aegis_data/burnin/burnin_const_R2000 \
        --release 200000
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#003366"
CRIMSON = "#8B2E2E"
ORANGE = "#CC6600"
GRAY = "#888888"


def survival_at(run_dir, AL, at_or_before=None, at_or_after=None):
    """Mean survival schedule from the phenotype snapshot nearest the requested step."""
    snaps = sorted((run_dir / "snapshots" / "phenotypes").glob("*.feather"),
                   key=lambda p: int(p.stem))
    if at_or_before is not None:
        cands = [s for s in snaps if int(s.stem) <= at_or_before] or snaps[:1]
        snap = cands[-1]
    else:
        cands = [s for s in snaps if int(s.stem) >= at_or_after] or snaps[-1:]
        snap = cands[-1]
    ph = pd.read_feather(snap)
    surv = ph[[f"surv_{a}" for a in range(AL)]].values
    return int(snap.stem), len(ph), surv.mean(axis=0), surv.std(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--release", type=int, required=True,
                   help="step at which regulation was released (phase-1 -> phase-2 boundary)")
    p.add_argument("-o", "--out", default="runs/burnin_and_release.png")
    p.add_argument("--zoom", type=int, default=6000,
                   help="show this many steps on either side of the release, so the "
                        "oscillation is legible instead of crushed against a long burn-in")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.run_dir / "final_config.yml"))
    AL = int(cfg["AGE_LIMIT"])
    mat = int(cfg["MATURATION_AGE"])
    ages = np.arange(AL)

    # survival at the release (end of burn-in) and at the end of the released phase
    pre_step, _, px_pre, sd_pre = survival_at(args.run_dir, AL, at_or_before=args.release)
    post_step, _, px_post, _ = survival_at(args.run_dir, AL, at_or_after=args.release + 1)

    # N(t) and R(t)
    n = pd.read_csv(args.run_dir / "popsize_before_reproduction.csv", header=None).squeeze("columns").to_numpy()
    r = pd.read_csv(args.run_dir / "resources_before_scavenging.csv", header=None).squeeze("columns").to_numpy()
    m = min(len(n), len(r))
    n, r = n[:m], r[:m]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    # LEFT — survival before (navy) and after release (orange), each with survivorship
    axL.plot(ages, px_pre, color=NAVY, lw=2, label=f"Survival — at release (step {pre_step:,})")
    axL.fill_between(ages, px_pre - sd_pre, px_pre + sd_pre, color=NAVY, alpha=0.10, lw=0)
    axL.plot(ages, px_pre.cumprod(), color=NAVY, lw=1.4, ls="--", alpha=0.7)
    axL.plot(ages, px_post, color=ORANGE, lw=2, label=f"Survival — released end (step {post_step:,})")
    axL.plot(ages, px_post.cumprod(), color=ORANGE, lw=1.4, ls="--", alpha=0.7)
    axL.axhline(0.95, color=GRAY, lw=1, ls=":", alpha=0.8)
    axL.annotate("non-aging start (0.95)", xy=(AL - 1, 0.95), xytext=(-4, 4),
                 textcoords="offset points", ha="right", fontsize=8, color=GRAY)
    axL.axvline(mat, color=GRAY, lw=1, ls=":", alpha=0.6)
    axL.annotate("maturation", xy=(mat, 0.02), xytext=(3, 0), textcoords="offset points",
                 fontsize=8, color=GRAY, rotation=90, va="bottom")
    axL.plot([], [], color=GRAY, lw=1.4, ls="--", label="survivorship $l_x=\\prod p_x$")
    axL.set_xlabel("Age class")
    axL.set_ylabel("Probability")
    axL.set_xlim(0, AL - 1)
    axL.set_ylim(0, 1.02)
    axL.set_title("Evolved survival: at release vs after resource stress", fontsize=10)
    axL.legend(frameon=False, fontsize=8, loc="lower left")
    axL.spines[["top", "right"]].set_visible(False)

    # stats over the full series (before windowing for the plot)
    reg = n[:args.release]
    osc = n[args.release:]

    # RIGHT — N(t) and R(t), windowed around the release; twin axis since R spikes high
    lo = max(0, args.release - args.zoom)
    hi = min(m, args.release + args.zoom)
    xw, nw, rw = np.arange(lo, hi), n[lo:hi], r[lo:hi]
    axRr = axR.twinx()
    axRr.plot(xw, rw, color=CRIMSON, lw=0.5, alpha=0.5)
    axRr.set_ylabel("Resources R", color=CRIMSON)
    axRr.tick_params(axis="y", colors=CRIMSON, labelsize=8)
    rfin = rw[np.isfinite(rw)]
    axRr.set_ylim(0, (np.nanmax(rfin) if len(rfin) else 1) * 1.05)
    axRr.spines[["top"]].set_visible(False)

    axR.plot(xw, nw, color=NAVY, lw=0.6)
    axR.set_zorder(axRr.get_zorder() + 1)
    axR.patch.set_visible(False)
    axR.set_ylim(0, np.nanmax(nw) * 1.05)
    axR.set_xlim(lo, hi)
    axR.set_xlabel("Simulation step")
    axR.set_ylabel("Population N", color=NAVY)
    axR.tick_params(axis="y", colors=NAVY, labelsize=8)
    axR.axvline(args.release, color="black", lw=1.2, ls="--")
    axR.annotate("regulation released\nresource dynamics on", xy=(args.release, np.nanmax(nw)),
                 xytext=(6, -4), textcoords="offset points", fontsize=8, va="top")
    axR.set_title("Population and resources across the phase boundary", fontsize=10)
    axR.spines[["top"]].set_visible(False)

    sub = (f"regulated N = {reg[-min(len(reg),20000):].mean():.0f} ± "
           f"{reg[-min(len(reg),20000):].std():.0f}   ->   released CV(N) = "
           f"{osc.std()/osc.mean():.2f}, range {osc.min():.0f}-{osc.max():.0f}"
           ) if len(osc) else "released phase not started"
    fig.suptitle("Two-phase resource experiment: equilibrate, then release\n" + sub, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  survival early(0-9)  release {px_pre[:10].mean():.4f} -> released end {px_post[:10].mean():.4f}"
          f"  ({px_post[:10].mean()-px_pre[:10].mean():+.4f})")
    print(f"  survival late(30-49) release {px_pre[30:].mean():.4f} -> released end {px_post[30:].mean():.4f}"
          f"  ({px_post[30:].mean()-px_pre[30:].mean():+.4f})")


if __name__ == "__main__":
    main()
