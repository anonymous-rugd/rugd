These 5 files are the original PGFPlots sources for the data-driven figures. They are kept for
reference (and are a valid, fully working alternative) but are **not currently `\input` by
main.tex or supplementary.tex** — the paper now uses pre-rendered vector PDFs instead
(`paper/figures/render_figures.py` → `*.pdf`), because re-executing 6 TikZ/PGFPlots figures on
every one of the 3-4 full recompiles that cleveref/hyperref/xr-hyper/natbib require was pushing
total compile time past Overleaf's free-tier timeout.

If compile time stops being a constraint (e.g. compiling locally, or on a paid Overleaf plan),
these can be swapped back in by replacing the corresponding `\includegraphics{figures/X.pdf}`
call with `\resizebox{...}{!}{\input{figures/X.tex}}` (after moving the file back up a
directory) — the visual content is the same, produced from the same
`paper/figures/data/*.csv` files either way.
