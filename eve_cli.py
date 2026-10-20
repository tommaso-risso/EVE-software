#!/usr/bin/env python3
"""
eve_cli.py - An interactive shell for running the EVE pipeline one stage at a
time: load data, pick methods, tweak parameters, run a stage, look, repeat.

Usage:
    python eve_cli.py        # then type 'help' at the eve> prompt

Parameters live in params.yaml (repo root). It uses the same layout as
config_template.yaml, so the file you tune here can also be run in one go with
`python eve_run.py params.yaml`. You can change it with the 'set', 'use' and
'reset' commands, or edit it in any text editor: the shell re-reads it before
every command, so both ways stay in sync. Values stay in the file between
sessions, so the last values you used are the ones the next run starts from.
"""
import cmd
import difflib
import inspect
import os
import datetime
import pickle
import re
import shutil
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

# eve_run.py also imports every EVE routine module (CandidateFinding etc.), so
# the plugin metadata helpers below work exactly as they do for the script.
from eve_run import (
    GLOBAL_SETTINGS_KEYS,
    REPO_ROOT,
    ConfigError,
    build_global_settings,
    get_metadata,
    load_events,
    optional_param_specs,
    required_param_specs,
    resolve_method,
    save_localizations_thunderstorm,
)
from EventDistributions import eventDistributions
from TemporalFitting import timeFitting
from Utils import utils

PARAMS_PATH = os.path.join(REPO_ROOT, "params.yaml")
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config_template.yaml")

# Every params.yaml section that holds methods, and the plugin folder each one
# draws them from. The first four are the pipeline stages, in order; 'preview'
# holds the methods 'show' uses to draw a single candidate.
STAGES = {
    "finding": "CandidateFinding",
    "fitting": "CandidateFitting",
    "post_processing": "PostProcessing",
    "visualisation": "Visualisation",
    "preview": "CandidatePreview",
}
PIPELINE = ["finding", "fitting", "post_processing", "visualisation"]
LIST_SECTIONS = ("post_processing", "preview")  # sections holding an ordered list of methods
DEFAULT_PREVIEW = ["TwoDProjections.TwoDProjection", "CumulativeSum.CumulativeSum"]
INPUT_KEYS = ["file", "start_time_ms", "end_time_ms"]


# ---------------------------------------------------------------------------
# Plugin parameters: names, defaults, allowed values
# ---------------------------------------------------------------------------

def option_choices(base_name):
    """Some fitting parameters (dist_kwarg, time_kwarg) don't take a number but
    the name of a class, e.g. time_kwarg: LognormCDFFirstEvents_weighted. The
    metadata only names the base class ('TemporalFits'); every subclass of it
    in EventDistributions/TemporalFitting is a valid choice (this is how the GUI
    fills its drop-down menus, see Utils/utils.py)."""
    for module in (eventDistributions, timeFitting):
        base = getattr(module, base_name, None)
        if inspect.isclass(base):
            return {
                name: getattr(obj, "display_name", name)
                for name, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, base) and obj is not base
            }
    return {}


def int_params(module):
    """Names of the parameters a plugin converts with int(kwargs['name']).
    The metadata can't tell us this: a default of 10 may be read with int() or
    with float() depending on the plugin, and int('3.5') crashes the run."""
    return set(re.findall(r"\bint\(\s*kwargs\[['\"](\w+)['\"]\]\s*\)", inspect.getsource(module)))


def param_specs(meta, module=None):
    """Every parameter a method accepts: {name: {description, default, required, choices?, kind?}}."""
    specs = {}
    for spec in required_param_specs(meta):
        specs[spec["name"]] = dict(spec, required=True)
    for spec in optional_param_specs(meta):
        specs[spec["name"]] = dict(spec, required=False)
    for key, value in meta.items():
        if isinstance(value, dict) and "base" in value and key in specs:
            specs[key]["choices"] = option_choices(value["base"])
    if module is not None:
        for name in int_params(module) & set(specs):
            specs[name]["kind"] = "int"
    return specs


def plugin_defaults(dotted_name):
    """The plugin's own default value for every parameter of a method."""
    _, _, meta = get_metadata(dotted_name)
    return {name: spec.get("default") for name, spec in param_specs(meta).items()}


def value_kind(default):
    """Guess what type a parameter expects from its default value. The metadata
    has no type field; defaults are written either as Python values (150., 30,
    False) or as strings ("0", "False"), so both forms are recognised."""
    if isinstance(default, bool) or str(default) in ("True", "False", "true", "false"):
        return "bool"
    try:
        float(default)
        return "number"
    except (TypeError, ValueError):
        return "str"


def parse_value(text, spec):
    """Turn what the user typed into a value of the type the parameter expects,
    or raise ValueError explaining why it doesn't fit."""
    choices = spec.get("choices")
    if choices:
        if text in choices:
            return text
        close = difflib.get_close_matches(text, choices, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"'{text}' is not a valid choice.{hint} Options: {', '.join(choices)}")

    kind = spec.get("kind") or value_kind(spec.get("default"))
    if kind == "bool":
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        raise ValueError(f"expected true or false, got '{text}'.")
    if kind in ("int", "number"):
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"expected a number, got '{text}'.")
        if kind == "int" and not number.is_integer():
            raise ValueError(f"expected a whole number (the method reads it with int()), got '{text}'.")
        return int(number) if kind == "int" or text.lstrip("-").isdigit() else number
    return text


def n_valid(localizations):
    """Number of real localizations. Fitting keeps a row for every candidate,
    with NaN coordinates where the fit was discarded, so len() over-counts."""
    return int(localizations["x"].notna().sum())


def suggest(word, options):
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" Did you mean '{close[0]}'?" if close else ""


# ---------------------------------------------------------------------------
# params.yaml
# ---------------------------------------------------------------------------

def read_params():
    if not os.path.exists(PARAMS_PATH):
        shutil.copy(TEMPLATE_PATH, PARAMS_PATH)
        print(f"Created {os.path.relpath(PARAMS_PATH)} from config_template.yaml.")
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f) or {}
    if "preview" not in params:  # config_template.yaml has no preview section
        params["preview"] = [{"method": name, "parameters": plugin_defaults(name)} for name in DEFAULT_PREVIEW]
    for stage in STAGES:
        if stage in LIST_SECTIONS:
            params[stage] = params.get(stage) or []
        else:
            params.setdefault(stage, {})
    return params


def write_params(params):
    with open(PARAMS_PATH, "w") as f:
        f.write("# eve_cli working parameters. Edit freely, or use 'set'/'use'/'reset' in eve_cli.\n"
                "# Same layout as config_template.yaml (see there for comments), so this file\n"
                "# also runs with: python eve_run.py params.yaml\n\n")
        yaml.safe_dump(params, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def stage_methods(params, stage):
    """[(section_dict, context_label), ...] for a stage: one entry for finding,
    fitting and visualisation, one per step for the post_processing and preview lists."""
    if stage in LIST_SECTIONS:
        return [(step, f"{stage}.{i}") for i, step in enumerate(params[stage])]
    return [(params[stage], stage)]


def stage_settings(params, stage):
    """The method(s) and parameters a stage would run with, in a form that can
    be compared to tell whether its settings changed since it last ran."""
    return [(section.get("method"), section.get("parameters") or {})
            for section, _ in stage_methods(params, stage)]


# ---------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------

class EveShell(cmd.Cmd):
    intro = ("\nEVE interactive shell. Type 'help' for the list of commands, "
             "'help <command>' for details, 'quit' to leave.\n"
             f"Parameters are stored in {os.path.relpath(PARAMS_PATH)}.\n")
    prompt = "eve> "

    def __init__(self):
        super().__init__()
        self.events = None
        self.events_file = None
        self.finding_result = None
        self.localizations = None        # output of fitting
        self.post_localizations = None   # output of the post_processing list
        self.image = None
        self.last_run = {}               # stage -> {"settings", "summary"} of its last run
        self.session_dir = None

    # --- shell plumbing ---------------------------------------------------

    def _pipeline_line(self):
        """The line above the prompt: each stage, whether it ran, and with what.
        ✓ ran with the current settings, ● ran but its settings changed since, ○ not run."""
        try:
            params = read_params()
        except Exception as e:  # e.g. a typo made while editing params.yaml by hand
            return f"(params.yaml can't be read: {e})"
        if self.events is None:
            parts = ["load ○"]
        else:
            parts = [f"load ✓ {os.path.splitext(os.path.basename(self.events_file))[0]}"]
        short = {"post_processing": "post", "visualisation": "vis"}
        for stage in PIPELINE:
            methods = "+".join(method.split(".", 1)[-1] for method, _ in stage_settings(params, stage) if method)
            last = self.last_run.get(stage)
            if last is None:
                mark, summary = "○", ""
            else:
                mark = "✓" if last["settings"] == stage_settings(params, stage) else "●"
                summary = f" ({last['summary']})"
            parts.append(f"{short.get(stage, stage)} {mark} {methods or '-'}{summary}")
        return " → ".join(parts)

    def _update_prompt(self):
        self.prompt = f"\n{self._pipeline_line()}\neve> "

    def postcmd(self, stop, line):
        self._update_prompt()
        return stop

    def preloop(self):
        # macOS ships a readline replacement (libedit) that needs a different
        # command to make the Tab key complete.
        try:
            import readline
            if "libedit" in (readline.__doc__ or ""):
                readline.parse_and_bind("bind ^I rl_complete")
        except ImportError:
            pass
        read_params()
        self._update_prompt()

    def onecmd(self, line):
        # One bad command (typo, plugin error) must not end the whole session.
        try:
            return super().onecmd(line)
        except ConfigError as e:
            print(str(e.code).strip())
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

    def emptyline(self):
        pass  # cmd's default would repeat the previous command, e.g. a long run

    def default(self, line):
        word = line.split()[0]
        print(f"Unknown command '{word}'.{suggest(word, self._command_names())} Type 'help' for the list.")

    def _command_names(self):
        return sorted(name[3:] for name in self.get_names()
                      if name.startswith("do_") and name not in ("do_EOF", "do_exit"))

    def do_help(self, arg):
        """help [command]: list all commands, or explain one in detail."""
        if arg:
            return super().do_help(arg)
        print("\nCommands (type 'help <command>' for details):\n")
        for name in self._command_names():
            doc = (getattr(self, f"do_{name}").__doc__ or "").strip().splitlines()
            print(f"  {doc[0] if doc else name}")
        print()

    def do_quit(self, arg):
        """quit: leave the shell. Parameters are already saved in params.yaml."""
        return True

    do_exit = do_quit

    def do_EOF(self, arg):
        print()
        return True

    # --- helpers ------------------------------------------------------------

    def _parse_stage(self, word, allowed=STAGES):
        if word not in allowed:
            raise ConfigError(f"Unknown stage '{word}'.{suggest(word, allowed)} "
                              f"Choose from: {', '.join(allowed)}")
        return word

    def _complete_from(self, options, text):
        return [o for o in options if o.startswith(text)]

    def _all_methods(self, stage):
        return sorted(utils.functionNamesFromDir(STAGES[stage]))

    # --- info ---------------------------------------------------------------

    def do_info(self, arg):
        """info [stage [method]]: explain stages, the methods of a stage, or one method's parameters.

        info                      list the stages and the method each currently uses
        info fitting              list every fitting method (* = currently selected)
        info fitting GaussianFitting.LogGaussian2D
                                  explain that method and all its parameters,
                                  including the allowed options for choice parameters
        """
        words = arg.split()
        params = read_params()
        if not words:
            print("\nStages, in pipeline order, plus the 'preview' methods used by 'show' "
                  "(type 'info <stage>' for its methods):\n")
            for stage in STAGES:
                current = [section.get("method", "(none)") for section, _ in stage_methods(params, stage)]
                print(f"  {stage:<16} {', '.join(current) or '(no steps)'}")
            print("\nAlso in params.yaml: 'input' (data file, time range) and 'global_settings'.\n")
            return

        stage = self._parse_stage(words[0])
        current = {section.get("method") for section, _ in stage_methods(params, stage)}
        if len(words) == 1:
            print(f"\n{stage} methods (* = currently selected; 'info {stage} <method>' for details):\n")
            for dotted_name in self._all_methods(stage):
                try:
                    _, _, meta = get_metadata(dotted_name)
                except ConfigError:
                    continue
                mark = "*" if dotted_name in current else " "
                print(f" {mark} {dotted_name:<50} {meta.get('display_name', '')}")
            print()
            return

        dotted_name = words[1]
        if dotted_name not in self._all_methods(stage):
            raise ConfigError(f"'{dotted_name}' is not a {stage} method."
                              f"{suggest(dotted_name, self._all_methods(stage))}")
        _, _, meta = get_metadata(dotted_name)
        current_values = {}
        for section, _ in stage_methods(params, stage):
            if section.get("method") == dotted_name:
                current_values = section.get("parameters") or {}
        print(f"\n{dotted_name}: {meta.get('display_name', '')}")
        if meta.get("help_string"):
            print(f"  {meta['help_string']}")
        for name, spec in param_specs(meta).items():
            kind = "required" if spec["required"] else "optional"
            value = f"  current: {current_values[name]}" if name in current_values else ""
            print(f"\n  {name} ({kind}, default: {spec.get('default')}){value}")
            if spec.get("description"):
                print(f"      {spec['description']}")
            for choice, display in (spec.get("choices") or {}).items():
                print(f"        - {choice:<48} {display}")
        print()

    def complete_info(self, text, line, begidx, endidx):
        words = line.split()
        if len(words) == 1 or (len(words) == 2 and text):
            return self._complete_from(STAGES, text)
        if words[1] in STAGES:
            return self._complete_from(self._all_methods(words[1]), text)
        return []

    # --- load ---------------------------------------------------------------

    def do_load(self, arg):
        """load [file]: load event data (.hdf5 or .npy) into memory.

        With a file, it also becomes input.file in params.yaml. Without one,
        the file already in params.yaml is (re)loaded. The time range comes from
        input.start_time_ms / input.end_time_ms (change with 'set input.end_time_ms ...').
        Loading new data clears all results computed from the old data.
        """
        params = read_params()
        params.setdefault("input", {})
        if arg.strip():
            params["input"]["file"] = arg.strip()
            write_params(params)
        file_path = params["input"].get("file")
        if not file_path or not os.path.exists(file_path):
            raise ConfigError(f"Data file '{file_path}' not found.")
        t0 = time.time()
        self.events = load_events(params["input"])
        self.events_file = file_path
        self.finding_result = self.localizations = self.post_localizations = self.image = None
        self.last_run = {}
        print(f"Loaded {len(self.events)} events from '{file_path}' ({time.time() - t0:.1f}s).")

    # --- use / set / reset / params ----------------------------------------

    def do_use(self, arg):
        """use <stage> <method> [<method> ...]: choose the method for a stage.

        use fitting GaussianFitting.Gaussian2D
        use post_processing DriftCorr_AIM.DriftCorr_AIM Filter.Regular_filter
                    (post_processing takes an ordered list; 'use post_processing none' clears it)
        use preview TwoDProjections.TwoDProjection CumulativeSum.CumulativeSum
                    (the views 'show <ID>' draws, one window each)

        Parameters of a newly chosen method start at the plugin's defaults; a method
        that stays selected keeps its current values. 'info <stage>' lists the methods.
        """
        words = arg.split()
        if len(words) < 2:
            raise ConfigError("Usage: use <stage> <method>. Type 'info <stage>' to see the methods.")
        stage = self._parse_stage(words[0])
        methods = [] if words[1:] == ["none"] and stage in LIST_SECTIONS else words[1:]
        if stage not in LIST_SECTIONS and len(methods) != 1:
            raise ConfigError(f"'{stage}' takes exactly one method.")
        available = self._all_methods(stage)
        for dotted_name in methods:
            if dotted_name not in available:
                raise ConfigError(f"'{dotted_name}' is not a {stage} method.{suggest(dotted_name, available)}")

        params = read_params()
        previous = {section.get("method"): section for section, _ in stage_methods(params, stage)}
        sections = []
        for dotted_name in methods:
            if dotted_name in previous:
                sections.append(previous[dotted_name])
            else:
                sections.append({"method": dotted_name, "parameters": plugin_defaults(dotted_name)})
        if stage in LIST_SECTIONS:
            params[stage] = sections
        else:
            # Keep extra keys of the section (e.g. finding's save_candidates).
            params[stage] = dict(params[stage], **sections[0])
        write_params(params)
        print(f"{stage}: {', '.join(methods) or '(no steps)'}. Type 'params {stage}' to see its parameters.")

    def complete_use(self, text, line, begidx, endidx):
        words = line.split()
        if len(words) == 1 or (len(words) == 2 and text):
            return self._complete_from(STAGES, text)
        if words[1] in STAGES:
            return self._complete_from(self._all_methods(words[1]), text)
        return []

    def _settable_paths(self, params):
        """Every 'set' path valid right now, mapped to the spec used to check its value."""
        paths = {}
        for key in GLOBAL_SETTINGS_KEYS:
            paths[f"global_settings.{key}"] = {"default": (params.get("global_settings") or {}).get(key)}
        for key in INPUT_KEYS:
            default = (params.get("input") or {}).get(key, 0)
            paths[f"input.{key}"] = {"default": "" if key == "file" else default}
        for stage in STAGES:
            for section, label in stage_methods(params, stage):
                if not section.get("method"):
                    continue
                module, _, meta = get_metadata(section["method"])
                for name, spec in param_specs(meta, module).items():
                    paths[f"{label}.{name}"] = spec
        return paths

    def do_set(self, arg):
        """set <stage>.<parameter> <value>: change one parameter.

        set fitting.expected_width 120
        set fitting.time_kwarg LognormCDFAllEvents
        set finding.debug true
        set post_processing.0.number_bins 2000     (post_processing steps are numbered from 0)
        set global_settings.PixelSize_nm 80
        set input.end_time_ms 60000

        Parameter names are checked against the current method, and the value is
        checked against the type of the plugin's default (number, true/false, or
        one of a fixed list of choices). Use Tab to complete names.
        """
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            raise ConfigError("Usage: set <stage>.<parameter> <value>")
        path, text = parts[0], parts[1].strip()
        params = read_params()
        paths = self._settable_paths(params)
        if path not in paths:
            raise ConfigError(f"Unknown parameter '{path}' for the current methods.{suggest(path, paths)}")
        try:
            value = parse_value(text, paths[path])
        except ValueError as e:
            raise ConfigError(f"{path}: {e}")

        keys = path.split(".")
        if keys[0] in LIST_SECTIONS:
            target = params[keys[0]][int(keys[1])].setdefault("parameters", {})
        elif keys[0] in STAGES:
            target = params[keys[0]].setdefault("parameters", {})
        else:
            target = params.setdefault(keys[0], {})
        target[keys[-1]] = value
        write_params(params)
        print(f"{path} = {value}")

    def complete_set(self, text, line, begidx, endidx):
        if len(line.split()) > 2 or (len(line.split()) == 2 and not text):
            return []
        return self._complete_from(self._settable_paths(read_params()), text)

    def do_reset(self, arg):
        """reset [stage]: put a stage's parameters back to the plugin defaults (all stages if none given).

        The selected methods stay the same; only their parameter values are reset.
        input and global_settings are not touched.
        """
        params = read_params()
        stages = [self._parse_stage(arg.strip())] if arg.strip() else list(STAGES)
        for stage in stages:
            for section, _ in stage_methods(params, stage):
                if section.get("method"):
                    section["parameters"] = plugin_defaults(section["method"])
        write_params(params)
        print(f"Reset parameters of: {', '.join(stages)}.")

    def complete_reset(self, text, line, begidx, endidx):
        return self._complete_from(STAGES, text)

    def do_params(self, arg):
        """params [section]: print the current parameters (all of params.yaml, or one section)."""
        params = read_params()
        section = arg.strip()
        if section:
            if section not in params:
                raise ConfigError(f"No section '{section}' in params.yaml.{suggest(section, params)}")
            params = {section: params[section]}
        print(yaml.safe_dump(params, sort_keys=False, default_flow_style=False, allow_unicode=True))

    def complete_params(self, text, line, begidx, endidx):
        return self._complete_from(list(read_params()), text)

    # --- run ----------------------------------------------------------------

    def do_run(self, arg):
        """run <stage> [-y]: run one pipeline stage, after a quick review of its method and parameters.

        run finding           needs 'load' first
        run fitting           needs finding
        run post_processing   runs the whole ordered list on the fitting results
        run visualisation     uses the post-processed localizations if available,
                              otherwise the fitting ones, and shows the image

        Before running you get:
          1. the list of methods for the stage: type a number, or press Enter to keep the current one
          2. a table of ALL its parameters (optional ones too): press Enter to run,
             type '<name> <value>' to change one (checked like 'set'), or 'q' to cancel
        Add -y to skip both and run straight away with the current settings.

        Re-running a stage clears the results of the stages after it. Debug plots and
        plots made by the method open in windows: close them to get the prompt back.
        """
        words = arg.split()
        if not words:
            raise ConfigError("Usage: run <stage> [-y]. Stages: " + ", ".join(PIPELINE))
        stage = self._parse_stage(words[0], PIPELINE)
        for word in words[1:]:
            if word != "-y":
                raise ConfigError(f"Unknown option '{word}'. The only option is -y (skip the review).")
        self._check_inputs(stage)  # fail before the review, not after it
        if "-y" not in words[1:] and not self._review(stage):
            print("Cancelled; nothing was run. Choices made in the review are kept in params.yaml.")
            return
        self._run_stage(stage)

    def complete_run(self, text, line, begidx, endidx):
        return self._complete_from(PIPELINE, text)

    def _check_inputs(self, stage):
        missing = {
            "finding": (self.events, "No data loaded. Use 'load <file>' first."),
            "fitting": (self.finding_result, "No candidates in memory. Run finding first."),
            "post_processing": (self.localizations, "No localizations in memory. Run fitting first."),
            "visualisation": (self.localizations, "No localizations in memory. Run fitting first."),
        }[stage]
        if missing[0] is None:
            raise ConfigError(missing[1])

    def _ask(self, question):
        """input() that treats end-of-input (e.g. Ctrl-D) as 'q'."""
        try:
            return input(question).strip()
        except EOFError:
            print()
            return "q"

    def _review(self, stage):
        """The guided part of 'run': pick the method(s), then review every
        parameter. Returns False if the user cancels."""
        methods = []
        for dotted_name in self._all_methods(stage):
            try:
                _, _, meta = get_metadata(dotted_name)
            except ConfigError:
                continue
            methods.append((dotted_name, meta.get("display_name", "")))
        current = [section.get("method") for section, _ in stage_methods(read_params(), stage)]

        print(f"\n{stage} methods (* = current):")
        for i, (dotted_name, display) in enumerate(methods, start=1):
            mark = "*" if dotted_name in current else " "
            print(f" {mark}{i:>3}. {dotted_name:<48} {display}")
        if stage in LIST_SECTIONS:
            question = (f"Methods to run, in order (e.g. '2 5'); 0 = none "
                        f"[Enter = keep: {' '.join(current) or 'none'}]: ")
        else:
            question = f"Method number [Enter = keep {current[0]}]: "

        while True:
            answer = self._ask(question)
            if answer == "q":
                return False
            if answer == "":
                break
            try:
                numbers = [int(word) for word in answer.split()]
                if stage in LIST_SECTIONS and numbers == [0]:
                    chosen = []
                elif stage not in LIST_SECTIONS and len(numbers) != 1:
                    raise ValueError
                else:
                    chosen = [methods[n - 1][0] for n in numbers if n >= 1]
                    if len(chosen) != len(numbers):
                        raise ValueError
            except (ValueError, IndexError):
                print(f"  Type {'numbers' if stage in LIST_SECTIONS else 'a number'} from the list, "
                      f"Enter to keep the current choice, or q to cancel.")
                continue
            if chosen != current:
                self.do_use(f"{stage} {' '.join(chosen) or 'none'}")
            break

        while True:
            params = read_params()
            sections = stage_methods(params, stage)
            if not sections:
                print("  (no methods selected)")
            for section, label in sections:
                self._print_param_table(section, label)
            answer = self._ask("Enter = run, '<name> <value>' = change a value, q = cancel: ")
            if answer == "":
                return True
            if answer == "q":
                return False
            parts = answer.split(maxsplit=1)
            if len(parts) != 2:
                print("  To change a value type its name and the new value, e.g. 'expected_width 120'.")
                continue
            try:
                self.do_set(f"{stage}.{parts[0]} {parts[1]}")
            except ConfigError as e:
                print("  " + str(e.code).strip().replace(f"'{stage}.", "'"))

    def _print_param_table(self, section, label):
        """One method's parameters: current value (or the plugin default when
        params.yaml doesn't list it), and what each one means."""
        module, _, meta = get_metadata(section["method"])
        values = section.get("parameters") or {}
        prefix = label.split(".", 1)[1] + "." if "." in label else ""
        print(f"\n{label}: {section['method']} ({meta.get('display_name', '')})")
        for name, spec in param_specs(meta, module).items():
            value = f"{values[name]}" if name in values else f"{spec.get('default')} (default)"
            description = (spec.get("description") or "").strip()
            if len(description) > 70:
                description = description[:67] + "..."
            print(f"  {prefix + name:<28} {value:<32} {description}")
            if spec.get("choices"):
                print(f"  {'':<28} options: {', '.join(spec['choices'])}")
        print()

    def _run_stage(self, stage):
        params = read_params()
        global_settings = build_global_settings(params)

        if stage == "finding":
            func, kwargs = resolve_method(params["finding"], STAGES[stage], "finding")
            t0 = time.time()
            result, _ = func(self.events, global_settings, **kwargs)
            self.finding_result = result
            self.localizations = self.post_localizations = self.image = None
            summary = f"{len(result)} cand"
            print(f"Finding: {len(result)} candidates from {len(self.events)} events ({time.time() - t0:.1f}s).")

        elif stage == "fitting":
            func, kwargs = resolve_method(params["fitting"], STAGES[stage], "fitting")
            t0 = time.time()
            result, _ = func(self.finding_result, global_settings, **kwargs)
            self.localizations = result
            self.post_localizations = self.image = None
            summary = f"{n_valid(result)} locs"
            print(f"Fitting: {n_valid(result)} localizations from {len(self.finding_result)} candidates "
                  f"({time.time() - t0:.1f}s).")

        elif stage == "post_processing":
            steps = [(resolve_method(section, STAGES[stage], label), section["method"])
                     for section, label in stage_methods(params, stage)]
            if not steps:
                raise ConfigError("No post_processing steps. Add some with 'use post_processing <method> ...'.")
            localizations = self.localizations
            for (func, kwargs), dotted_name in steps:
                t0 = time.time()
                localizations, _ = func(localizations, self.finding_result, global_settings, **kwargs)
                print(f"Post-processing {dotted_name}: {n_valid(localizations)} localizations "
                      f"({time.time() - t0:.1f}s).")
            self.post_localizations = localizations
            self.image = None
            summary = f"{n_valid(localizations)} locs"

        else:  # visualisation
            localizations = self.post_localizations if self.post_localizations is not None else self.localizations
            source = "post-processed" if self.post_localizations is not None else "fitting"
            func, kwargs = resolve_method(params["visualisation"], STAGES[stage], "visualisation")
            t0 = time.time()
            image, scale = func(localizations, global_settings, **kwargs)
            self.image = image
            summary = f"of {'post' if source == 'post-processed' else 'fitting'}"
            print(f"Visualisation of {n_valid(localizations)} {source} localizations: "
                  f"image {getattr(image, 'shape', '?')}, scale {scale} ({time.time() - t0:.1f}s).")
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.set_title(f"{params['visualisation']['method']} ({source} localizations)")
            ax.axis("off")

        # Remember what this stage ran with (for the pipeline line above the
        # prompt); the later stages' results were just cleared.
        for later in PIPELINE[PIPELINE.index(stage) + 1:]:
            self.last_run.pop(later, None)
        self.last_run[stage] = {"settings": stage_settings(params, stage), "summary": summary}
        self._show_figures()

    def _show_figures(self):
        """Show every figure still open (the visualisation image, plots a method
        created without showing them). Blocks until the windows are closed."""
        if plt.get_fignums():
            print("  Showing plot window(s): close them to return to the prompt.")
            plt.show()
            plt.close("all")

    # --- show ---------------------------------------------------------------

    def _parse_candidate_id(self, word):
        if self.finding_result is None:
            raise ConfigError("No candidates in memory. Use 'run finding' first.")
        try:
            candidate_id = int(word)
        except ValueError:
            raise ConfigError(f"'{word}' is not a candidate ID (a whole number).")
        if candidate_id not in self.finding_result:
            raise ConfigError(f"No candidate {candidate_id}. IDs go from {min(self.finding_result)} "
                              f"to {max(self.finding_result)}.")
        return candidate_id

    def _draw_candidate(self, candidate_id):
        """Draw one candidate with every method in the 'preview' list, one figure
        each (the preview plugins lay out the whole figure themselves, so they
        can't share one). Returns [(method name, figure), ...]."""
        params = read_params()
        global_settings = build_global_settings(params)
        previews = [(resolve_method(section, STAGES["preview"], label), section["method"])
                    for section, label in stage_methods(params, "preview")]
        if not previews:
            raise ConfigError("No preview methods. Add some with 'use preview <method> ...' "
                              "(see 'info preview').")

        candidate = self.finding_result[candidate_id]
        # The fit belongs to fitting, not post-processing: drift correction moves
        # the localization away from the candidate's own (uncorrected) events.
        if self.localizations is not None:
            fit = self.localizations[self.localizations["candidate_id"] == candidate_id]
        else:
            fit = pd.DataFrame(columns=["x", "y", "p", "t", "candidate_id"])

        if fit.empty:
            fit_status = "not fitted"
        elif fit["x"].isna().all():
            fit_status = "fit discarded"
        else:
            fit_status = ", ".join(f"x={row.x:.1f} nm y={row.y:.1f} nm t={row.t:.1f} ms"
                                   for row in fit.itertuples())
        print(f"Candidate {candidate_id}: {candidate['N_events']} events, "
              f"cluster size {list(candidate['cluster_size'])}, {fit_status}.")

        figures = []
        for (func, kwargs), dotted_name in previews:
            fig = plt.figure(figsize=(8, 6))
            fig.canvas.manager.set_window_title(f"Candidate {candidate_id}: {dotted_name}")
            try:
                func(candidate["events"], fit, self.events, fig, global_settings, **kwargs)
            except Exception as e:
                # e.g. a view that needs a fit, while fitting hasn't run yet
                plt.close(fig)
                hint = " (it may need 'run fitting' first)" if fit.empty else ""
                print(f"  {dotted_name} could not draw this candidate{hint}: {type(e).__name__}: {e}")
                continue
            figures.append((dotted_name, fig))
        if not figures:
            raise ConfigError("None of the preview methods could draw this candidate.")
        return figures

    def do_show(self, arg):
        """show <ID>: draw one candidate with each method in the 'preview' list.

        Uses the candidates from the last finding run and, if fitting ran, that
        candidate's fit. Choose the views with 'use preview ...' (see 'info preview')
        and tune them with 'set preview.0.<parameter> ...'. Close the windows to get
        the prompt back.
        """
        candidate_id = self._parse_candidate_id(arg.strip())
        self._draw_candidate(candidate_id)
        self._show_figures()

    # --- save / export ------------------------------------------------------

    def _session_dir(self):
        """One output folder per shell session, created on the first save/export."""
        if self.session_dir is None:
            base_dir = (read_params().get("output") or {}).get("base_dir", "output")
            self.session_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"Saving this session's files in {self.session_dir}/")
        return self.session_dir

    def _output_path(self, name, extension):
        """A new file path in the session folder, plus a copy of the current
        params.yaml next to it so the file can be traced to its parameters."""
        stamp = datetime.datetime.now().strftime("%H-%M-%S")
        base = os.path.join(self._session_dir(), f"{name}_{stamp}")
        n = 2
        while os.path.exists(base + extension):  # two saves within the same second
            base = os.path.join(self._session_dir(), f"{name}_{stamp}_{n}")
            n += 1
        shutil.copy(PARAMS_PATH, base + "_params.yaml")
        return base + extension

    def do_save(self, arg):
        """save image | save <ID>: save a picture into this session's output folder.

        save image    the last visualisation image, as .tif
        save 42       candidate 42 drawn with the 'preview' methods, as a PDF (one page per method)

        Every saved file gets a copy of params.yaml next to it (<name>_params.yaml).
        """
        word = arg.strip()
        if not word:
            raise ConfigError("Usage: save image | save <ID>")
        if word == "image":
            if self.image is None:
                raise ConfigError("No image in memory. Use 'run visualisation' first.")
            path = self._output_path("visualisation", ".tif")
            imageio.imwrite(path, np.asarray(self.image, dtype=np.float32))
        else:
            candidate_id = self._parse_candidate_id(word)
            figures = self._draw_candidate(candidate_id)
            path = self._output_path(f"candidate_{candidate_id}", ".pdf")
            with PdfPages(path) as pdf:
                for _, fig in figures:
                    pdf.savefig(fig)
            plt.close("all")
        print(f"Saved {path}")

    def complete_save(self, text, line, begidx, endidx):
        return self._complete_from(["image"], text)

    def do_export(self, arg):
        """export <stage>: write the data from a stage into this session's output folder.

        export finding           the candidates, as a Python pickle
        export fitting           the fitting localizations, as a ThunderSTORM CSV
        export post_processing   the post-processed localizations, as a ThunderSTORM CSV

        Discarded fits (no coordinates) are left out of the CSVs. Every exported
        file gets a copy of params.yaml next to it (<name>_params.yaml).
        """
        stage = self._parse_stage(arg.strip(), ["finding", "fitting", "post_processing"])
        data = {"finding": self.finding_result, "fitting": self.localizations,
                "post_processing": self.post_localizations}[stage]
        if data is None:
            raise ConfigError(f"No {stage} results in memory. Use 'run {stage}' first.")
        if stage == "finding":
            path = self._output_path("candidates", ".pickle")
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            path = self._output_path(f"{stage}_localizations", ".csv")
            save_localizations_thunderstorm(path, data)
        print(f"Exported {path}")

    def complete_export(self, text, line, begidx, endidx):
        return self._complete_from(["finding", "fitting", "post_processing"], text)


def main():
    os.chdir(REPO_ROOT)  # relative paths in params.yaml are relative to the EVE folder
    EveShell().cmdloop()


if __name__ == "__main__":
    main()
