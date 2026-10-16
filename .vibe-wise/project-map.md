# Project Map

## Purpose
EVE: PyQt5 GUI plus scriptable API for localizing emitters in event-based SMLM data (verified from CLAUDE.md).

## Requirements
Not specified yet (no task given).

## Components
- GUI.py: launcher around MyGUI in GUI_main.py (~5.4k lines; run/preview orchestration, settings save/load, multiprocessing).
- Plugin folders (CandidateFinding, CandidateFitting, PostProcessing, Visualisation, CandidatePreview, EventDistributions, TemporalFitting): each .py exposes __function_metadata__().
- Utils/utils.py: scans plugin folders, evals metadata; data loading helpers.
- Data/: example data and GUI settings JSON.

## Main Flow
event data -> Candidate Finding -> Candidate Fitting -> Post-processing / Visualisation (existing, per CLAUDE.md)
plugin metadata -> Utils/utils.py scan -> GUI_main.py builds widgets

## Data and Trust Boundaries
Loads .npy, .hdf5, Prophesee .raw (Metavision SDK). Other boundaries unknown.

## Build and Deployment
Python 3.9; conda env (environment_eve.yml) or pip -r requirements.txt; run `python GUI.py`. No tests/linter.

## Unknowns
Task for this session; learner's areas of focus.
