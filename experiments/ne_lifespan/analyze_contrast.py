"""Compare regimes: N_mean, Ne_demog (trough-driven), Ne_genetic (theta_w, units-correct),
and evolved intrinsic lifespan e0 from the genetic surv phenotype (not realized deaths)."""
import sys, statistics, pathlib
import pandas as pd, numpy as np, yaml


def per_step_census(d):
    ns = pd.read_csv(pathlib.Path(d) / "popsize_after_reproduction.csv", header=None)[0].values.astype(float)
    ns = ns[ns > 0]
    return ns


def genetic_ne(d):
    cfg = yaml.safe_load(open(pathlib.Path(d).with_suffix(".yml")))
    s = pd.read_csv(pathlib.Path(d) / "popgen" / "simple.csv").iloc[-1]
    g = pd.read_feather(sorted((pathlib.Path(d) / "snapshots" / "genotypes").glob("*.feather"),
                               key=lambda p: int(p.stem))[-1])
    ploidy = 2 if "mean_h" in s.index else 1
    L = g.shape[1] // ploidy                       # sites per haploid genome
    ne, theta, theta_w = s["ne"], s["theta"], s["theta_w"]
    # units: theta is per-site, theta_w is genome-total -> divide by L. Validated on smoke.
    ne_gen = ne * theta_w / (theta * L)
    mu = cfg["G_muta_initpheno"]
    ne_gen_check = theta_w / (L * 2 * ploidy * mu)  # exact form (theta in csv is 3-sig-fig rounded)
    assert abs(ne_gen - ne_gen_check) < 2e-2 * max(ne_gen, 1), (ne_gen, ne_gen_check)
    return ne_gen_check, theta_w / (theta * L)      # genetic Ne (exact), theta_w/theta_persite ratio


def evolved_lifespan(d):
    """e0 from the genetic surv phenotype: lx = cumprod(mean surv per age), e0 = sum(lx)."""
    p = pd.read_feather(sorted((pathlib.Path(d) / "snapshots" / "phenotypes").glob("*.feather"),
                               key=lambda p: int(p.stem))[-1])
    surv_cols = [c for c in p.columns if c.startswith("surv_")]
    surv_cols.sort(key=lambda c: int(c.split("_")[1]))
    px = p[surv_cols].mean(axis=0).values             # mean survival per age
    lx = np.cumprod(px)
    return float(px.mean()), float(lx.sum()), px      # mean px, life expectancy e0, curve


print(f"{'regime':<8}{'N_mean':>8}{'Ne_demog':>10}{'trough':>8}{'Ne_genetic':>12}{'thw/th':>8}"
      f"{'mean_px':>9}{'e0(life)':>9}")
curves = {}
for d in sys.argv[1:]:
    name = pathlib.Path(d).name
    ns = per_step_census(d)
    n_mean = statistics.mean(ns); ne_demog = statistics.harmonic_mean(ns); trough = min(ns)
    ne_gen, ratio = genetic_ne(d)
    mean_px, e0, px = evolved_lifespan(d)
    curves[name] = px
    print(f"{name:<8}{n_mean:>8.0f}{ne_demog:>10.0f}{trough:>8.0f}{ne_gen:>12.0f}{ratio:>8.3f}"
          f"{mean_px:>9.4f}{e0:>9.2f}")

if len(curves) == 2:
    a, b = list(curves)
    print(f"\nper-age survival px  (age: {a}  vs  {b}):")
    for age in range(0, len(curves[a]), max(1, len(curves[a]) // 15)):
        print(f"  age {age:>2}:  {curves[a][age]:.3f}   {curves[b][age]:.3f}")
