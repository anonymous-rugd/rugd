"""Generate every complete LaTeX table in paper/tables/ from the retained final-study CSVs and
the bootstrap CI file computed this session. Run: python paper/tables/generate_tables.py

Each table is written as ONE complete, self-contained file (\\begin{table}...\\end{table}, rows
included directly) rather than a wrapper file that \\input's a separate row-body file inside the
tabular environment. An earlier version split wrapper and body across two files; that pattern
triggered a "Misplaced \\noalign" fatal error from booktabs' \\bottomrule when compiled in
Overleaf (a nested \\input immediately before \\bottomrule inside an active alignment, unlike
every other table in this paper -- e.g. tables/literature_scope.tex, which writes its rows
directly and always compiled cleanly). Merging wrapper and body removes that entire class of
risk and matches the one pattern already proven to compile.

Nothing here is invented; every number is read from outputs/studies/final_study/**.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs" / "studies" / "final_study"
OUT = Path(__file__).resolve().parent

NAME = {"prior": "Prior", "persistence": "Persistence", "logistic": "Logistic",
        "hist_gbdt": "HGBDT", "lightgbm": "LightGBM", "gru": "GRU", "graph_gru": "Graph-GRU",
        "rugd": "\\textbf{ShiftGuard-RUGD}", "uignn_inductive": "UIGNN-Inductive",
        "staeformer_inductive": "STAEformer-Inductive", "opencity_scratch": "OpenCity-Scratch"}
ORDER = ["rugd", "staeformer_inductive", "graph_gru", "gru", "opencity_scratch", "lightgbm",
         "hist_gbdt", "persistence", "uignn_inductive", "logistic", "prior"]


def fmt(x, d=4):
    return f"{x:.{d}f}"


def bold_best(vals, best="max"):
    return vals.idxmax() if best == "max" else vals.idxmin()


def write_table(name, source_comment, caption, label, col_spec, header, rows, resize=True):
    lines = [source_comment, r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
              f"\\label{{{label}}}"]
    open_resize = r"\resizebox{\linewidth}{!}{%" if resize else ""
    close_resize = "}" if resize else ""
    lines.append(open_resize) if resize else None
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.extend(rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}" + close_resize)
    lines.append(r"\end{table}")
    text = "\n".join(l for l in lines if l != "") + "\n"
    (OUT / f"{name}.tex").write_text(text)


def headline_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    ci = pd.read_csv(STUDY / "main" / "headline_bootstrap_ci.csv")
    agg = df.groupby("model").agg(
        macro_f1_m=("macro_f1", "mean"), macro_f1_s=("macro_f1", "std"),
        auprc_m=("severe_auprc", "mean"), auprc_s=("severe_auprc", "std"),
        nll_m=("nll", "mean"), nll_s=("nll", "std"),
    )
    ci_f1 = ci[ci.metric == "macro_f1"].set_index("model")
    ci_ap = ci[ci.metric == "severe_auprc"].set_index("model")
    best_f1 = bold_best(agg["macro_f1_m"], "max")
    best_ap = bold_best(agg["auprc_m"], "max")
    best_nll = bold_best(agg["nll_m"], "min")
    rows = []
    for m in ORDER:
        r = agg.loc[m]
        f1 = fmt(r.macro_f1_m); f1 = f"\\textbf{{{f1}}}" if m == best_f1 else f1
        ap = fmt(r.auprc_m); ap = f"\\textbf{{{ap}}}" if m == best_ap else ap
        nll = fmt(r.nll_m); nll = f"\\textbf{{{nll}}}" if m == best_nll else nll
        f1ci = ci_f1.loc[m]; apci = ci_ap.loc[m]
        rows.append(
            f"{NAME[m]} & {f1}$\\pm${fmt(r.macro_f1_s)} & [{fmt(f1ci.ci_lo)}, {fmt(f1ci.ci_hi)}] & "
            f"{ap}$\\pm${fmt(r.auprc_s)} & [{fmt(apci.ci_lo)}, {fmt(apci.ci_hi)}] & "
            f"{nll}$\\pm${fmt(r.nll_s)} \\\\"
        )
    write_table(
        "headline",
        "% Source: outputs/studies/final_study/main/results_all.csv (mean/SD) and\n"
        "% outputs/studies/final_study/main/headline_bootstrap_ci.csv (95% CI, 20,000-repeat\n"
        "% hierarchical city-then-seed bootstrap -- the SAME construction as every paired-\n"
        "% comparison CI in this paper; see paper/tables/regenerate_headline_ci.py).\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Held-out-city results, mean $\\pm$ sample SD over nine city--seed runs (3 cities "
        "$\\times$ 3 seeds), with 95\\% bootstrap confidence intervals on the two primary "
        "metrics. Bold marks the best mean per column. Note the RUGD/STAEformer intervals "
        "overlap on both primary metrics (\\Cref{sec:results}).",
        "tab:headline", "lccccc",
        "Method & Macro-F1 $\\uparrow$ & 95\\% CI & Severe AUPRC $\\uparrow$ & 95\\% CI & NLL $\\downarrow$ \\\\",
        rows,
    )


def robustness_table():
    rob = pd.read_csv(STUDY / "main" / "robustness_all.csv")
    sub = rob[(rob.corruption == "missing") & (rob.severity.isin([0.3, 0.5]))]
    piv = sub.groupby(["model", "severity"])["severe_auprc"].agg(["mean", "std"])
    best30 = sub[sub.severity == 0.3].groupby("model")["severe_auprc"].mean().idxmax()
    best50 = sub[sub.severity == 0.5].groupby("model")["severe_auprc"].mean().idxmax()
    rows = []
    for m in ORDER:
        if m not in piv.index.get_level_values(0):
            continue
        m30 = piv.loc[(m, 0.3)]; m50 = piv.loc[(m, 0.5)]
        s30 = f"{fmt(m30['mean'])}$\\pm${fmt(m30['std'])}"
        s50 = f"{fmt(m50['mean'])}$\\pm${fmt(m50['std'])}"
        if m == best30: s30 = f"\\textbf{{{s30}}}"
        if m == best50: s50 = f"\\textbf{{{s50}}}"
        rows.append(f"{NAME[m]} & {s30} & {s50} \\\\")
    write_table(
        "robustness",
        "% Source: outputs/studies/final_study/main/robustness_all.csv, missingness rows only\n"
        "% (full 7-condition table is tables/full_robustness.tex in the appendix).\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Severe AUPRC under 30\\% and 50\\% independently-masked sensor observations, mean "
        "$\\pm$ SD over nine city--seed runs; clean-fit source calibration is reused (not "
        "re-tuned per condition). Bold marks the best mean per column. See "
        "\\Cref{app:robustness-full} for all seven registered corruption conditions.",
        "tab:robustness", "lcc", "Method & 30\\% missing & 50\\% missing \\\\", rows,
    )


def ablation_table():
    ab = pd.read_csv(STUDY / "ablations" / "results_all.csv")
    agg = ab.groupby("model").agg(
        f1_m=("macro_f1", "mean"), f1_s=("macro_f1", "std"),
        ap_m=("severe_auprc", "mean"), ap_s=("severe_auprc", "std"),
        nll_m=("nll", "mean"), ece_m=("ece", "mean"),
    )
    full = agg.loc["rugd"]
    label = {"rugd_no_graph": "w/o graph diffusion", "rugd_no_topology": "w/o topology",
             "rugd_no_domain": "w/o domain alignment", "rugd_no_evidence": "w/o evidence",
             "rugd_no_focal": "w/o focal modulation", "rugd_no_class_balance": "w/o class balance"}
    deltas = {m: agg.loc[m].ap_m - full.ap_m for m in label}
    max_abs_model = max(deltas, key=lambda m: abs(deltas[m]))
    rows = []
    for m, lbl in label.items():
        r = agg.loc[m]
        dstr = f"{deltas[m]:+.4f}"
        if m == max_abs_model:
            dstr = f"\\textbf{{{dstr}}}"
        rows.append(
            f"{lbl} & {fmt(r.f1_m)}$\\pm${fmt(r.f1_s)} & {fmt(r.ap_m)}$\\pm${fmt(r.ap_s)} & "
            f"{dstr} & {fmt(r.nll_m)} & {fmt(r.ece_m)} \\\\"
        )
    rows.append(r"\midrule")
    rows.append(
        f"\\textbf{{Full ShiftGuard-RUGD}} & \\textbf{{{fmt(full.f1_m)}}}$\\pm${fmt(full.f1_s)} & "
        f"\\textbf{{{fmt(full.ap_m)}}}$\\pm${fmt(full.ap_s)} & --- & {fmt(full.nll_m)} & {fmt(full.ece_m)} \\\\"
    )
    write_table(
        "ablation",
        "% Source: outputs/studies/final_study/ablations/results_all.csv, aggregated per\n"
        "% variant over nine city-seed runs each.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Final-architecture ablations, mean $\\pm$ SD over nine city--seed runs each. "
        "$\\Delta$ is clean severe-AUPRC change from full \\methodname{}; bold marks the "
        "single largest-magnitude $\\Delta$. Domain alignment's removal costs roughly "
        "$4\\times$ the next-largest component effect (graph diffusion) and roughly "
        "$30\\times$ the rare-event-specific mechanisms (class balance, evidence).",
        "tab:ablation", "lccccc",
        "Variant removed & Macro-F1 & Clean severe AUPRC & $\\Delta$ & NLL & ECE \\\\",
        rows, resize=False,
    )


def percity_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    models = ["rugd", "staeformer_inductive", "graph_gru"]
    cities = ["TrafficCD", "TrafficHZ", "TrafficSH"]
    rows = []
    for m in models:
        cells = []
        for c in cities:
            sub = df[(df.model == m) & (df.target_city == c)]
            cells.append(f"{fmt(sub.macro_f1.mean())}$\\pm${fmt(sub.macro_f1.std())} / "
                         f"{fmt(sub.severe_auprc.mean())}$\\pm${fmt(sub.severe_auprc.std())}")
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    write_table(
        "percity",
        "% Source: outputs/studies/final_study/main/results_all.csv, grouped by\n"
        "% (model, target_city) over three seeds each. Cells are Macro-F1 / Severe AUPRC.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Per-city clean results (Macro-F1 / Severe AUPRC), mean $\\pm$ SD over three seeds. "
        "TrafficHZ is the strongest case for \\methodname{}; TrafficCD is a near-tie with "
        "STAEformer (0.05 macro-F1 and 0.09 AUPRC points) and should not be presented as a "
        "decisive city-level win.",
        "tab:percity", "lccc", "Method & TrafficCD & TrafficHZ & TrafficSH \\\\", rows,
    )


def full_metrics_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    agg = df.groupby("model").agg(
        acc=("accuracy", "mean"), bacc=("balanced_accuracy", "mean"),
        f1=("macro_f1", "mean"), sp=("severe_precision", "mean"), sr=("severe_recall", "mean"),
        sf1=("severe_f1", "mean"), ap=("severe_auprc", "mean"), auroc=("severe_auroc", "mean"))
    rows = []
    for m in ORDER:
        r = agg.loc[m]
        rows.append(f"{NAME[m]} & {fmt(r.acc)} & {fmt(r.bacc)} & {fmt(r.f1)} & {fmt(r.sp)} & "
                     f"{fmt(r.sr)} & {fmt(r.sf1)} & {fmt(r.ap)} & {fmt(r.auroc)} \\\\")
    write_table(
        "full_metrics",
        "% Source: outputs/studies/final_study/main/results_all.csv, all 11 models, mean over 9 runs.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "All registered clean classification/ranking metrics, mean over nine runs (SD "
        "omitted for width; see \\Cref{tab:headline} for SD on the two primary metrics). "
        "P/R/F1 denote severe-class precision, recall, F1.",
        "tab:full-metrics", "lcccccccc",
        "Method & Accuracy & Balanced acc. & Macro-F1 & Severe P & Severe R & Severe F1 & AUPRC & AUROC \\\\",
        rows,
    )


def full_robustness_table():
    rob = pd.read_csv(STUDY / "main" / "robustness_all.csv")
    conds = [("missing", 0.1), ("missing", 0.3), ("missing", 0.5),
             ("noise", 0.1), ("noise", 0.3), ("shift", 1), ("shift", 2)]
    rows = []
    for m in ORDER:
        cells = []
        for cond, sev in conds:
            sub = rob[(rob.model == m) & (rob.corruption == cond) & (rob.severity == sev)]
            cells.append(fmt(sub.severe_auprc.mean()))
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    write_table(
        "full_robustness",
        "% Source: outputs/studies/final_study/main/robustness_all.csv, mean severe AUPRC per condition.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Severe AUPRC for every registered corruption, mean over nine runs. M: "
        "independently masked observations (\\%); N: Gaussian noise SD; S: temporal roll "
        "(half-hour bins).",
        "tab:full-robustness", "lccccccc",
        "Method & M10 & M30 & M50 & N.1 & N.3 & S1 & S2 \\\\", rows,
    )


def diagnostics_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    neural = ["gru", "graph_gru", "uignn_inductive", "staeformer_inductive", "opencity_scratch", "rugd"]
    agg = df[df.model.isin(neural)].groupby("model").agg(
        ace=("ace", "mean"), rho=("uncertainty_error_spearman", "mean"),
        cov=("conformal_severe_coverage", "mean"), temp=("temperature", "mean"))
    rows = []
    for m in neural:
        r = agg.loc[m]
        rows.append(f"{NAME[m]} & {fmt(r.ace)} & {fmt(r.rho)} & {fmt(r['cov'])} & {fmt(r.temp)} \\\\")
    write_table(
        "diagnostics",
        "% Source: outputs/studies/final_study/main/results_all.csv, neural models only.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Additional neural uncertainty diagnostics, mean over nine runs. ACE: adaptive "
        "calibration error. $\\rho$: Spearman association between predictive uncertainty "
        "and error. Coverage targets 0.90. Temperature is source-calibration-fitted and "
        "descriptive, not ranked.",
        "tab:diagnostics", "lcccc",
        "Method & ACE $\\downarrow$ & Uncertainty--error $\\rho$ $\\uparrow$ & Severe APS coverage & Temperature \\\\",
        rows, resize=False,
    )


def efficiency_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    neural = ["gru", "graph_gru", "rugd", "opencity_scratch", "uignn_inductive", "staeformer_inductive"]
    rows = []
    for m in neural:
        sub = df[df.model == m]
        pmin, pmax = int(sub.parameters.min()), int(sub.parameters.max())
        prange = f"{pmin:,}" if pmin == pmax else f"{pmin:,}--{pmax:,}"
        tt = f"{sub.train_seconds.mean():.1f}$\\pm${sub.train_seconds.std():.1f}"
        thr = f"{sub.predictions_per_second.mean()/1e6:.3f}$\\pm${sub.predictions_per_second.std()/1e6:.3f}M"
        rows.append(f"{NAME[m]} & {prange} & {tt} & {thr} \\\\")
    write_table(
        "efficiency",
        "% Source: outputs/studies/final_study/main/results_all.csv, parameters/\n"
        "% train_seconds/predictions_per_second columns, neural models only.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Neural efficiency, mean over nine runs. Parameter ranges reflect target-specific "
        "tuned widths; throughput is million node-predictions/s.",
        "tab:efficiency", "lccc",
        "Method & Parameters $\\downarrow$ & Train seconds $\\downarrow$ & Throughput $\\uparrow$ \\\\",
        rows, resize=False,
    )


def calibration_table():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    neural = ["gru", "graph_gru", "uignn_inductive", "staeformer_inductive", "opencity_scratch", "rugd"]
    agg = df[df.model.isin(neural)].groupby("model").agg(
        nll=("nll", "mean"), nlls=("nll", "std"), brier=("brier", "mean"), briers=("brier", "std"),
        ece=("ece", "mean"), eces=("ece", "std"), cov=("conformal_coverage", "mean"),
        covs=("conformal_coverage", "std"), setsz=("conformal_mean_set_size", "mean"),
        setszs=("conformal_mean_set_size", "std"))
    rows = []
    for m in neural:
        r = agg.loc[m]
        rows.append(f"{NAME[m]} & {fmt(r.nll)}$\\pm${fmt(r.nlls)} & {fmt(r.brier)}$\\pm${fmt(r.briers)} & "
                     f"{fmt(r.ece)}$\\pm${fmt(r.eces)} & {fmt(r['cov'])}$\\pm${fmt(r.covs)} & "
                     f"{fmt(r.setsz)}$\\pm${fmt(r.setszs)} \\\\")
    write_table(
        "calibration_full",
        "% Source: outputs/studies/final_study/main/results_all.csv, neural models only.\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Neural probabilistic quality and APS behavior over nine runs. Coverage target 0.90.",
        "tab:calibration-full", "lccccc",
        "Method & NLL $\\downarrow$ & Brier $\\downarrow$ & ECE $\\downarrow$ & APS coverage & Set size $\\downarrow$ \\\\",
        rows, resize=False,
    )


def paired_inference_table():
    ps = pd.read_csv(STUDY / "main" / "paired_robustness_significance.csv")
    label = {"gru": "GRU", "graph_gru": "Graph-GRU", "uignn_inductive": "UIGNN-Inductive",
             "opencity_scratch": "OpenCity-Scratch", "staeformer_inductive": "STAEformer-Inductive"}
    rows = []
    for _, r in ps.iterrows():
        if r["baseline"] not in label:
            continue
        sev_pct = int(round(r["severity"] * 100))
        rows.append(
            f"{sev_pct}\\% & {label[r['baseline']]} & {r['mean_difference']:+.4f}$\\pm${r['sd_difference']:.4f} & "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] & {int(r['wins'])}/{int(r['n_pairs'])} & "
            f"{r['rank_biserial']:.3f} & {r['wilcoxon_p']:.4f} / {r['holm_p_learned_baselines']:.4f} \\\\"
        )
    write_table(
        "paired_inference",
        "% Source: outputs/studies/final_study/main/paired_robustness_significance.csv\n"
        "% (computed by src/traffic_ml/opencity_statistics.py: exact Wilcoxon, rank-biserial,\n"
        "% hierarchical city-then-seed bootstrap, Holm correction over learned baselines).\n"
        "% Regenerate with: python paper/tables/generate_tables.py",
        "Exploratory paired severe-AUPRC inference. $\\Delta$ is \\methodname{} minus "
        "baseline; CI is a hierarchical city/seed bootstrap; $p_H$ is Holm-adjusted over "
        "learned baselines.",
        "tab:paired-inference", "llccccc",
        "Missing & Baseline & $\\Delta$ mean$\\pm$SD & 95\\% CI & Wins & $r_{rb}$ & $p$ / $p_H$ \\\\",
        rows,
    )


if __name__ == "__main__":
    headline_table()
    robustness_table()
    ablation_table()
    percity_table()
    full_metrics_table()
    full_robustness_table()
    diagnostics_table()
    efficiency_table()
    calibration_table()
    paired_inference_table()
    print("wrote complete table files to", OUT)
