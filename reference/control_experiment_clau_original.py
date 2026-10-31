"""Control experiment: is the portfolio better than plain random search / hyper-parameter-only search at equal budget?

Variants (same splits, same baselines, same promotion guard, same number of generations/proposals):
  portfolio : random + evolution + novelty + guided, curriculum A -> B -> D
  random_ABD: random strategy only, same curriculum (same search space, no evolution)
  hpo_A     : random strategy only, level A only (pure hyper-parameter search of the default recipe)
Usage: python scripts/control_experiment.py [--gens 25] [--seeds 1 2 3] [--out reports]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amde.config import Config  # noqa: E402
from amde.evaluation.evaluator import EvalResult  # noqa: E402
from amde.runtime.engine import Engine  # noqa: E402
from amde.runtime.resources import ResourcePolicy  # noqa: E402

VARIANTS = {
    "portfolio": dict(),
    "random_ABD": dict(strategies=["random"]),
    "hpo_A": dict(strategies=["random"], levels=["A"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        for name, over in VARIANTS.items():
            cfg = Config.smoke(seed=seed, challenge_every_gens=0, name=f"ctl-{name}", **over)
            with Engine(cfg, tempfile.mkdtemp(prefix=f"ctl_{name}_"), policy=ResourcePolicy(pause_max_s=60, poll_s=1), console=False) as e:
                e.run(max_generations=a.gens)
                st = e.state
                base = e.baselines[("adam_mlp16", 2)]
                row = {"variant": name, "seed": seed, "trials": st.trials_total, "baseline_dev": base.agg("dev"), "baseline_search": base.agg("search"),
                       "champion": bool(st.champion)}
                if st.champion_eval:
                    r = EvalResult.from_dict(st.champion_eval)
                    row.update(champion_dev=r.agg("dev"), champion_search=r.agg("search"), champion_worst=r.worst_family(),
                               champion_kind=st.champion.genome.model["kind"], champion_level=st.champion.level,
                               rule_nodes=len(json.dumps(st.champion.genome.rule)))
                rows.append(row)
                print(json.dumps(row), flush=True)
    summ = {}
    for name in VARIANTS:
        rs = [r for r in rows if r["variant"] == name]
        d = [r.get("champion_dev", r["baseline_dev"]) for r in rs]
        summ[name] = {"mean_champion_dev": sum(d) / len(d), "per_seed": d, "n_with_champion": sum(r["champion"] for r in rs),
                      "mean_trials": sum(r["trials"] for r in rs) / len(rs)}
    out = {"gens": a.gens, "seeds": a.seeds, "profile": "Config.smoke (tiny budgets: max_steps=90)", "summary": summ, "rows": rows,
           "note": "champion_dev = fid2 dev-family score of the promoted champion (dev was used for promotion in every variant); "
                   "baseline_dev is adam_mlp16 default. Runs without a champion are scored at the baseline."}
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "control_experiment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
