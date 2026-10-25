# CSET485 AI and Society – Assignment 1

- **Roll number:** S24CSEU1192
- **Dataset:** ProPublica COMPAS, `compas-scores-two-years.csv` (https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv). A copy is in `data/`, and the notebook downloads it if it is missing.
- **Split:** 70/30 train/test, `random_state = 192`.

## Contents
| File | Purpose |
|---|---|
| `analysis.ipynb` | Full analysis, executed with outputs: preprocessing, Part 1, Part 2.1–2.4. Writes `results.json` and `figures/`. |
| `analysis_src.py` / `make_nb.py` | Source of the notebook cells and the script that builds the notebook from them. |
| `build_report.py` | Builds `report.pdf` (HTML → PDF with WeasyPrint). Every number in it comes from `results.json`. |
| `report.pdf` | Final report: preprocessing, Parts 1–4. |
| `results.json`, `wordcounts.json` | Computed results and word counts. |
| `figures/` | Charts used in the report. |
| `data/` | The COMPAS CSV and the 50-row adversarial sample. |

## How to run
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # runs top to bottom, writes results.json + figures
python build_report.py                                                # writes report.pdf
```
WeasyPrint needs the Pango system libraries (`apt install libpango-1.0-0 libpangoft2-1.0-0` on Debian/Ubuntu).
