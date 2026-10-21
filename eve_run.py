#!/usr/bin/env python3
"""
eve_run.py - Run the EVE pipeline (Candidate Finding -> Candidate Fitting ->
Post-processing -> Visualisation) from the terminal, driven by a YAML config,
and save every result in a timestamped, self-describing output folder.

Usage:
    python eve_run.py --list-config      # list every available method and its parameters
    python eve_run.py my_config.yaml     # run the pipeline

See config_template.yaml for a commented starting point.
"""
import argparse
import csv
import datetime
import logging
import os
import pickle
import subprocess
import sys
import time

# The non-interactive "Agg" backend is selected in main(), not here, so that
# eve_cli.py can import the helpers in this file and still open plot windows.
import matplotlib
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import numpy as np
import yaml

# Same import pattern as EVE_CommandLine.ipynb: these wildcard imports put
# every routine module (e.g. EigenFeatureAnalysis, GaussianFitting) directly
# into this file's namespace, as EVE's own plugin-discovery code expects.
from CandidateFinding import *
from CandidateFitting import *
from PostProcessing import *
from Visualisation import *
from CandidatePreview import *  # not used by this script; lets eve_cli.py resolve preview methods via get_metadata
from Utils import utils, utilsHelper

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Global settings that CandidateFinding/CandidateFitting/PostProcessing
# routines actually read (found by grepping every routine for settings['...']).
# GUI-only keys (window size, storage paths, logging path, ...) are left out
# on purpose: this script never touches the GUI.
GLOBAL_SETTINGS_KEYS = [
    "PixelSize_nm",
    "MinFindingBoundingBoxXY",
    "MinFindingBoundingBoxT",
    "MaxFindingBoundingBoxXY",
    "MaxFindingBoundingBoxT",
    "Multithread",
    "UseCUDA",
]

FOLDER_LABELS = {
    "CandidateFinding": "Finding methods",
    "CandidateFitting": "Fitting methods",
    "PostProcessing": "Post-processing methods (used as an ordered list)",
    "Visualisation": "Visualisation methods",
}


class ConfigError(SystemExit):
    """A problem in the user's YAML config. Raised as SystemExit so the
    script stops immediately with a clear message, before any computation."""

    def __init__(self, message):
        super().__init__(f"\nConfig error: {message}\n")


# ---------------------------------------------------------------------------
# Reading EVE's plugin metadata (no GUI involved)
# ---------------------------------------------------------------------------

def get_metadata(dotted_name):
    """Resolve 'ModuleName.function_name' (e.g. 'EigenFeatureAnalysis.eigenFeature_analysis')
    to (module, function, metadata_entry), using the same __function_metadata__()
    contract GUI_main.py relies on."""
    if "." not in dotted_name:
        raise ConfigError(
            f"'{dotted_name}' is not a valid method name. "
            f"Expected 'Module.function', e.g. 'EigenFeatureAnalysis.eigenFeature_analysis'."
        )
    module_name, func_name = dotted_name.split(".", 1)
    module = globals().get(module_name)
    if module is None:
        raise ConfigError(f"Unknown module '{module_name}' (from '{dotted_name}').")
    try:
        metadata = module.__function_metadata__()
    except AttributeError:
        raise ConfigError(f"Module '{module_name}' has no __function_metadata__().")
    if func_name not in metadata:
        raise ConfigError(f"Unknown method '{func_name}' in module '{module_name}'.")
    if not hasattr(module, func_name):
        raise ConfigError(f"'{func_name}' is listed in {module_name}'s metadata but the function itself is missing.")
    return module, getattr(module, func_name), metadata[func_name]


def required_param_specs(metadata_entry):
    """EVE's metadata lists most required parameters under 'required_kwargs'. Some
    fitting routines additionally read extra kwargs directly (e.g. 'dist_kwarg',
    'time_kwarg' for the event-distribution and time-fitting choice) without going
    through that list. Those are functionally required too - the routine raises a
    bare KeyError if they're missing - so this flattens both into one list of
    {'name', 'description', 'default'} specs."""
    specs = list(metadata_entry.get("required_kwargs", []))
    for key, value in metadata_entry.items():
        if key in ("required_kwargs", "optional_kwargs", "help_string", "display_name"):
            continue
        if isinstance(value, dict) and "base" in value:
            specs.append({
                "name": key,
                "description": value.get("description", ""),
                "default": value.get("default_option"),
            })
    return specs


def optional_param_specs(metadata_entry):
    return list(metadata_entry.get("optional_kwargs", []))


# ---------------------------------------------------------------------------
# --list-config
# ---------------------------------------------------------------------------

def print_method_tree(folder):
    print(f"\n{FOLDER_LABELS[folder]}")
    print("=" * len(FOLDER_LABELS[folder]))
    for dotted_name in sorted(utils.functionNamesFromDir(folder)):
        try:
            _, _, meta = get_metadata(dotted_name)
        except ConfigError:
            continue
        print(f"\n  {dotted_name}")
        print(f"    {meta.get('display_name', '')}")
        help_string = meta.get("help_string", "")
        if help_string:
            print(f"    {help_string}")
        required = required_param_specs(meta)
        if required:
            print("    required parameters:")
            for spec in required:
                default = spec.get("default")
                hint = f" (suggested: {default})" if default is not None else ""
                print(f"      - {spec['name']}: {spec.get('description', '')}{hint}")
        optional = optional_param_specs(meta)
        if optional:
            print("    optional parameters (EVE's own default is used if omitted):")
            for spec in optional:
                print(f"      - {spec['name']}: {spec.get('description', '')} (default: {spec.get('default')})")


def list_config():
    print("Every method below can be used in a config file as 'ModuleName.function_name'.")
    for folder in ("CandidateFinding", "CandidateFitting", "PostProcessing", "Visualisation"):
        print_method_tree(folder)


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    if not config:
        raise ConfigError(f"'{path}' is empty.")
    return config


def require_keys(d, keys, context):
    missing = [key for key in keys if key not in d]
    if missing:
        raise ConfigError(f"Missing {', '.join(missing)} in {context}.")


def build_global_settings(config):
    """Expand the flat 'global_settings' section of the YAML into the nested
    {'Key': {'value': ...}} shape every EVE routine expects."""
    raw = config.get("global_settings", {}) or {}
    require_keys(raw, GLOBAL_SETTINGS_KEYS, "'global_settings'")
    return {key: {"value": raw[key]} for key in GLOBAL_SETTINGS_KEYS}


def build_kwargs(parameters, metadata_entry, context):
    """Turn the config's 'parameters' into the exact string kwargs the EVE
    routine needs (see EVE_CommandLine.ipynb, e.g. linearity_cutoff="0.7").

    Required parameters must be present, or the script stops here with an
    error. Optional parameters fall back to the default from the routine's
    own metadata when omitted: some routines (e.g. EigenFeatureAnalysis's
    'debug') read an "optional" kwarg directly from the kwargs dict without
    applying that default themselves, so it must be filled in here instead."""
    parameters = dict(parameters or {})

    required_specs = required_param_specs(metadata_entry)
    missing = [spec["name"] for spec in required_specs if spec["name"] not in parameters]
    if missing:
        raise ConfigError(
            f"Missing required parameter(s) {missing} for {context}.\n"
            f"Run 'python eve_run.py --list-config' to see every method and its parameters."
        )

    for spec in optional_param_specs(metadata_entry):
        parameters.setdefault(spec["name"], spec.get("default"))

    return {name: str(value) for name, value in parameters.items()}


def resolve_method(section_config, folder, context):
    dotted_name = (section_config or {}).get("method")
    if not dotted_name:
        raise ConfigError(f"'{context}' is missing a 'method'.")
    module, func, meta = get_metadata(dotted_name)
    kwargs = build_kwargs(section_config.get("parameters"), meta, f"{context} ('{dotted_name}')")
    return func, kwargs


POLARITIES = ("mix", "pos", "neg", "both")  # as in the GUI's polarity dropdown
POLARITY_VALUE = {"pos": 1, "neg": 0}       # the value of the events' 'p' field


def get_polarity(input_config):
    polarity = str((input_config or {}).get("polarity", "both")).lower()
    if polarity not in POLARITIES:
        raise ConfigError(f"input.polarity must be one of {', '.join(POLARITIES)}, not '{polarity}'.")
    return polarity


def load_events(input_config):
    """Load the events. 'pos' / 'neg' keep only that polarity; 'mix' and 'both' keep
    all of them ('both' is split later, so finding runs on each polarity on its own)."""
    require_keys(input_config or {}, ["file"], "'input'")
    file_path = input_config["file"]
    start_ms = input_config.get("start_time_ms", 0)
    end_ms = input_config.get("end_time_ms", float("inf"))
    polarity = get_polarity(input_config)
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".hdf5", ".h5"):
        events = utils.timeSliceFromHDF(
            file_path, requested_start_time_ms=start_ms, requested_end_time_ms=end_ms
        )[0]
    elif ext == ".npy":
        events = np.load(file_path)
    else:
        raise ConfigError(f"Unsupported input file type '{ext}' for '{file_path}'. Use .hdf5 or .npy.")
    if polarity in POLARITY_VALUE:
        events = events[events["p"] == POLARITY_VALUE[polarity]]
    return events


def run_finding(func, events, polarity, global_settings, kwargs):
    """Call a finding routine. With polarity 'both' it runs on the positive and on the
    negative events separately (same parameters), as the GUI's 'Pos and neg separately'
    does, and merges the candidates: the negative ones are numbered after the positive ones.
    Returns (candidates, info) like the routine itself."""
    if polarity != "both":
        return func(events, global_settings, **kwargs)
    outer = utilsHelper.get_progress_listener()
    merged, infos = {}, {}
    for i, name in enumerate(("pos", "neg")):
        part = events[events["p"] == POLARITY_VALUE[name]]
        if len(part) == 0:
            logging.warning(f"No {'positive' if name == 'pos' else 'negative'} events to run finding on.")
            continue
        if outer is not None:  # each polarity fills half of the progress bar
            utilsHelper.set_progress_listener(lambda fraction, i=i: outer((i + fraction) / 2))
        try:
            candidates, infos[name] = func(part, global_settings, **kwargs)
        finally:
            utilsHelper.set_progress_listener(outer)
        offset = len(merged)
        for key, candidate in candidates.items():
            merged[key + offset] = candidate
    return merged, infos


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_localizations_thunderstorm(path, localizations):
    """Write localizations in the same ThunderSTORM CSV format the GUI writes
    (see GUI_main.py's storeLocalization), so results can be opened by other
    SMLM tools, not only by EVE."""
    localizations = localizations.copy().dropna()
    localizations = localizations.applymap(lambda x: "1" if x == "" else x)
    localizations["frame"] = localizations["t"].apply(round).astype(int)
    localizations["frame"] -= min(localizations["frame"]) - 1

    rename = {
        "x": '"x [nm]"', "y": '"y [nm]"', "z": '"z [nm]"', "t": '"t [ms]"',
        "del_x": "Δx [nm]", "del_y": "Δy [nm]", "del_t": "Δt [ms]",
        "x_dim": "x_dim [px]", "y_dim": "y_dim [px]", "t_dim": "t_dim [ms]",
    }
    headers = [rename.get(h, h) for h in localizations.columns]
    localizations.to_csv(
        path, header=headers, quoting=csv.QUOTE_NONE, index=False, index_label='"id"'
    )


def save_new_figures_as_pdf(existing_fignums, directory):
    """Save and close every matplotlib figure opened since existing_fignums was
    captured. Post-processing routines call plt.show() when visualisation is on;
    under the Agg backend that does nothing, so figures are saved here instead."""
    new_fignums = [n for n in plt.get_fignums() if n not in existing_fignums]
    for i, n in enumerate(new_fignums):
        fig = plt.figure(n)
        suffix = "" if len(new_fignums) == 1 else f"_{i + 1}"
        fig.savefig(os.path.join(directory, f"plot{suffix}.pdf"))
        plt.close(fig)


def to_plain(value):
    """Convert numpy scalars/arrays - e.g. the image scale Visualisation routines
    return - into plain Python types, so PyYAML can write metadata.yaml."""
    if isinstance(value, dict):
        return {key: to_plain(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------

def run(config_path):
    config = load_config(config_path)

    global_settings = build_global_settings(config)
    events = load_events(config.get("input", {}))
    print(f"Loaded {len(events)} events from '{config['input']['file']}'.")

    # Resolve and validate every stage's method and parameters up front, so a
    # mistake in a later stage (e.g. post-processing step 2) is reported
    # before any computation runs, not after finding/fitting already ran.
    finding_config = config.get("finding", {})
    finding_func, finding_kwargs = resolve_method(finding_config, "CandidateFinding", "finding")
    save_candidates = bool(finding_config.get("save_candidates", False))

    fitting_config = config.get("fitting", {})
    fitting_func, fitting_kwargs = resolve_method(fitting_config, "CandidateFitting", "fitting")

    postprocessing_steps = []
    for i, step_config in enumerate(config.get("post_processing", []) or []):
        func, kwargs = resolve_method(step_config, "PostProcessing", f"post_processing[{i}]")
        postprocessing_steps.append((step_config, func, kwargs))

    vis_config = config.get("visualisation", {})
    vis_func, vis_kwargs = resolve_method(vis_config, "Visualisation", "visualisation")

    # --- set up the run folder ---
    output_base = (config.get("output") or {}).get("base_dir", "output")
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_base, run_id)
    finding_dir = os.path.join(run_dir, "finding")
    fitting_dir = os.path.join(run_dir, "fitting")
    postproc_dir = os.path.join(run_dir, "post_processing")
    vis_dir = os.path.join(run_dir, "visualization")
    for directory in (finding_dir, fitting_dir, postproc_dir, vis_dir):
        os.makedirs(directory, exist_ok=True)
    print(f"Run folder: {run_dir}")

    run_metadata = {
        "config": config,
        "git_commit": get_git_commit(),
        "n_events": int(len(events)),
        "stages": {},
    }

    # --- Finding ---
    t0 = time.time()
    finding_result, finding_info = run_finding(
        finding_func, events, get_polarity(config.get("input", {})), global_settings, finding_kwargs)
    run_metadata["stages"]["finding"] = {
        "method": finding_config["method"],
        "parameters": finding_config.get("parameters", {}),
        "duration_s": round(time.time() - t0, 2),
        "n_candidates": len(finding_result),
        "info": str(finding_info),
    }
    print(f"Finding complete: {len(finding_result)} candidates "
          f"({run_metadata['stages']['finding']['duration_s']}s)")
    if save_candidates:
        with open(os.path.join(finding_dir, "candidates.pickle"), "wb") as f:
            pickle.dump(finding_result, f)

    # --- Fitting ---
    t0 = time.time()
    localizations, fitting_info = fitting_func(finding_result, global_settings, **fitting_kwargs)
    run_metadata["stages"]["fitting"] = {
        "method": fitting_config["method"],
        "parameters": fitting_config.get("parameters", {}),
        "duration_s": round(time.time() - t0, 2),
        "n_localizations": len(localizations),
        "info": str(fitting_info),
    }
    print(f"Fitting complete: {len(localizations)} localizations "
          f"({run_metadata['stages']['fitting']['duration_s']}s)")
    save_localizations_thunderstorm(os.path.join(fitting_dir, "localizations.csv"), localizations)

    # --- Post-processing (an ordered list; each step feeds the next) ---
    run_metadata["stages"]["post_processing"] = []
    for step_config, func, kwargs in postprocessing_steps:
        method_name = step_config["method"].split(".", 1)[1]
        step_dir = os.path.join(postproc_dir, method_name)
        os.makedirs(step_dir, exist_ok=True)

        existing_fignums = set(plt.get_fignums())
        t0 = time.time()
        result = func(localizations, finding_result, global_settings, **kwargs)
        if result is None:  # steps that only draw a plot (Histogramming, NeNA, ...) return None: no change, as in the GUI
            step_info = "returned nothing: the localizations are unchanged"
        else:
            localizations, step_info = result
        duration = round(time.time() - t0, 2)
        save_new_figures_as_pdf(existing_fignums, step_dir)
        save_localizations_thunderstorm(os.path.join(step_dir, "localizations.csv"), localizations)

        run_metadata["stages"]["post_processing"].append({
            "method": step_config["method"],
            "parameters": step_config.get("parameters", {}),
            "duration_s": duration,
            "n_localizations": len(localizations),
            "info": str(step_info),
        })
        print(f"Post-processing '{method_name}' complete: {len(localizations)} localizations ({duration}s)")

    # --- Visualisation ---
    t0 = time.time()
    image, scale = vis_func(localizations, global_settings, **vis_kwargs)
    duration = round(time.time() - t0, 2)
    imageio.imwrite(os.path.join(vis_dir, "image.tif"), np.asarray(image, dtype=np.float32))
    run_metadata["stages"]["visualisation"] = {
        "method": vis_config["method"],
        "parameters": vis_config.get("parameters", {}),
        "duration_s": duration,
        "scale": scale,
    }
    print(f"Visualisation complete ({duration}s). Image scale: {scale}")

    # --- Metadata: everything needed to know how this run was produced ---
    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(to_plain(run_metadata), f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"\nRun complete. Results saved to: {run_dir}")


def main():
    # Force a non-interactive matplotlib backend before any figure is created.
    # Some PostProcessing routines call plt.show(), which would otherwise open
    # a window and block the script.
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Run the EVE pipeline from the terminal, driven by a YAML config."
    )
    parser.add_argument("config", nargs="?", help="Path to a YAML config file (see config_template.yaml).")
    parser.add_argument(
        "--list-config", action="store_true",
        help="List every available method and its parameters, grouped by stage, then exit.",
    )
    args = parser.parse_args()

    if args.list_config:
        list_config()
        return

    if not args.config:
        parser.error("a config file is required unless --list-config is given")

    run(args.config)


if __name__ == "__main__":
    main()
