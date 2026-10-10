import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"joy": "#1f77b4", "imu": "#d62728", "keyboard": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(HERE, "results"))
    ap.add_argument("--crit", type=float, default=1.5, help="숙달 기준 평균 오차(m)")
    ap.add_argument("--exclude", default="SELFTEST,ZERO,TEST")
    a = ap.parse_args()
    excl = set(a.exclude.split(","))
    rows = [r for r in csv.DictReader(open(os.path.join(a.res, "results.csv"), encoding="utf-8"))
            if r["pid"] not in excl and r["input"] != "geom"]
    if not rows:
        raise SystemExit("결과 없음")
    runs = defaultdict(dict)
    for r in rows:
        runs[(r["input"], r["pid"])][int(r["trial"])] = r

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    summ = []
    by_input = defaultdict(list)
    for (inp, pid), tr in sorted(runs.items()):
        ks = sorted(tr)
        off = [float(tr[k]["mean_abs_offset_m"]) for k in ks]
        ontr = [float(tr[k]["on_track_pct"]) for k in ks]
        ax[0].plot(ks, off, "-o", color=COLORS.get(inp, "k"), alpha=0.35, lw=1, ms=3)
        ax[1].plot(ks, ontr, "-o", color=COLORS.get(inp, "k"), alpha=0.35, lw=1, ms=3)
        by_input[inp].append((ks, off, ontr))
        learned, cum = None, 0.0
        for k in ks:
            cum += float(tr[k]["seconds"])
            if float(tr[k]["mean_abs_offset_m"]) < a.crit and int(tr[k]["off_track_count"]) == 0:
                learned = (k, cum)
                break
        summ.append(dict(input=inp, pid=pid, trials=len(ks), first_offset_m=off[0], last_offset_m=off[-1],
                         first_on_track_pct=ontr[0], last_on_track_pct=ontr[-1],
                         learned_trial=learned[0] if learned else "", learned_after_s=learned[1] if learned else ""))
    for inp, lst in by_input.items():
        n = max(len(k) for k, _, _ in lst)
        for j, axi in ((1, 0), (2, 1)):
            M = np.full((len(lst), n), np.nan)
            for i, item in enumerate(lst):
                M[i, :len(item[j])] = item[j]
            ax[axi].plot(range(1, n + 1), np.nanmean(M, 0), "-o", color=COLORS.get(inp, "k"), lw=2.5,
                         label=f"{inp} (n={len(lst)})")
    ax[0].axhline(a.crit, ls="--", color="k", lw=0.8)
    ax[0].set(xlabel="trial", ylabel="mean |lane offset| (m)", title="accuracy (lower = better)")
    ax[1].set(xlabel="trial", ylabel="time on track (%)", title="on-track ratio", ylim=(0, 102))
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.res, "learning_curve.png"), dpi=150)
    with open(os.path.join(a.res, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
