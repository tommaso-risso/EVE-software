# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EVE: PyQt5 GUI (plus a scriptable API) for localizing emitters in event-based SMLM (eveSMLM) data. Requires Python 3.9. There is no test suite and no linter config.

## Commands

- Setup: `conda env create -f environment_eve.yml` or `pip install -r requirements.txt`
- Run GUI: `python GUI.py` (a thin launcher around `MyGUI` in `GUI_main.py`; the packaged entry point is `eve-SMLM`)
- Headless usage: `EVE_CommandLine.ipynb` shows the full load → find → fit → post-process → visualise pipeline. It does `from CandidateFinding import *`, etc. from the repo root.
- Example data and GUI configs (`GUI_Settings_*.json`) live in `Data/` (Nanoruler, Ecoli, aTubulin). Load them in the GUI via `Settings -> Load specific GUI contents` to try a change end to end.
- PyPI release is a manual copy-the-repo-into-`src/eve_smlm` process, described in `PyPI/PyPI_INSTRUCTIONS.txt`. That is why `GUI.py` tries `from eve_smlm.GUI_main import ...` first and falls back to `GUI_main`; keep imports working in both layouts.

## Architecture

Pipeline: **Candidate Finding** (event list → clusters) → **Candidate Fitting** (clusters → x,y,(z),t,p localizations) → **Post-processing / Visualisation**. Candidate Preview and TemporalFitting support inspection and lifetime fitting.

**Plugin discovery drives everything.** Each routine folder (`CandidateFinding`, `CandidateFitting`, `PostProcessing`, `Visualisation`, `CandidatePreview`, `EventDistributions`, `TemporalFitting`) holds `.py` files that expose `__function_metadata__()`. It returns a dict of function name → `required_kwargs` / `optional_kwargs` / `help_string` / `display_name`, and a function of the same name must exist in the file. `Utils/utils.py` scans these folders, `eval`s the metadata, and `GUI_main.py` builds the GUI widgets from it. So adding a routine means dropping in a file with the right metadata, with no GUI edits. Keep the metadata and the function signature in sync, or the GUI will break or warn.

Routine I/O contracts (the full spec is in `DeveloperInstructions.md`):
- Finding: `f(npy_array, settings, **kwargs) -> (candidates, performance_metadata)`. The input is a structured array with fields `x,y,p,t`, with t in µs. `candidates` is a dict of `{events: DataFrame(x,y,t,p), N_events, cluster_size}`.
- Fitting works on candidates and returns localizations plus metadata.
- Post-processing returns changed localizations or None. Visualisation returns an image array plus its scale. Preview routines draw into a figure.

**`GUI_main.py`** (~5k lines) is the monolith. It holds the run/preview orchestration (`run_processing`, `previewRun`, `runFinding`, `runFitting`, with polarity-specific variants, `runFindingOnPolarity` and `runFittingOnPolarity`), GUI state save/load to JSON (`load_entries_from_json_single`), and multiprocessing. Polarity mode ("Pos and neg separately" vs combined) changes which widgets exist, so changes to settings handling must account for both modes.

`Utils/` has data loading (`.npy`, `.hdf5`, Prophesee `.raw` needing the Metavision SDK) and shared helpers.

## Gotchas

- Drift correction by entropy minimization (DME, `PostProcessing/dme`) and the Gaussian visualisations use pre-compiled Windows DLLs, so they only run on Windows. Everything else is cross-platform. `PostProcessing/DriftCorr_AIM.py` (with `aim/`) is the pure-Python alternative.
- `Utils/utils.py` has a hardcoded Windows user path for an extra plugin folder (`...AppData\Local\UniBonn\Eve`). Don't copy that pattern.
- Docs: `README.md` (user manual), `DeveloperInstructions.md` (plugin API), `ScientificBackground.md` (algorithm descriptions).
