#!/usr/bin/env python3
"""
Bounded hyperparameter study (docs/TRAINING.md).

    python scripts/tune.py --base_run runs/my_model --study_dir runs/tune_study
    python scripts/tune.py --study_dir runs/tune_study --report
    python scripts/tune.py --score_runs runs/my_model_s45 runs/my_tuned_model_s45 ...

A trial is the base run's recipe with six settings changed (SPACE below): learning rate, weight
decay, dropout, label smoothing, focal gamma and hidden sizes. Everything else comes from the base
run's config.json: training pickle, label map, exclusions, groups, classes, read simulation,
dilution, encoding, early stopping, calibration and seed, so every trial uses the same folds and is
scored on the same simulated inputs. Each trial runs the full nested cross-validation of
scripts/train.py in this process, saves no network weights, and is scored on its out-of-fold
predictions. Only training arrays are read; nanopore files play no part.

Objective: the mean recall over classes (balanced accuracy), averaged with equal weight over the
binary-read conditions at 10, 30 and 50% coverage, each undiluted and at 50% and 30% blasts.
Temperature scaling does not change which class ranks first, so calibration does not affect it.

The first trial of a study is the base run's own settings, so the study contains its baseline.
Several jobs can work on one study at a time: it is a journal file in --study_dir on the shared
disk, and every job stops starting trials once --max_trials trials have finished or been pruned.

Pruning: during fold 1, the accuracy on the inner validation split (the split that already drives
early stopping) is reported after every epoch. A trial stops when it is below the median of the
finished trials at the same epoch, once --startup_trials trials have finished and after
--warmup_epochs epochs. The outer folds play no part. --no_pruning turns it off.

Outputs in --study_dir:
  study.journal          the Optuna study
  space.json             search space, objective and base run; written once, later jobs must match
  trials/trial_<n>/      settings.json, metrics.csv (every scored condition), recall_by_class.csv
                         (objective conditions) and folds_summary.csv
  trials.csv             all trials, best first (--report)

--score_runs gives the same objective, and the callable share at 98% accuracy of calls, for
finished scripts/train.py runs (from their cv_predictions_<condition>.csv files), so confirmation
runs on new seeds are scored exactly as the trials were. Runs whose folder names differ only in a
final _s<seed> are also averaged together.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from compare_runs import callable_at_accuracy  # noqa: E402
from data.dataset import encode_labels, filter_classes, load_label_map, load_training_data, subset  # noqa: E402
from evaluation.conditions import condition_name  # noqa: E402
from evaluation.metrics import summarize_probs  # noqa: E402
from models.dilution import normal_index  # noqa: E402
from models.sparse_nn import softmax_np  # noqa: E402
from training.reproducibility import set_deterministic_mode  # noqa: E402
from training.trainer import TrainConfig, cross_validate  # noqa: E402

logger = logging.getLogger("sparsh")

STUDY_NAME = "sparsh_tune"
# The search space, fixed before the study was run. Change it only for a new study.
SPACE = {
    "learning_rate": {"type": "float", "low": 3e-5, "high": 3e-4, "log": True},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "dropout": {"type": "float", "low": 0.2, "high": 0.6},
    "label_smoothing": {"type": "float", "low": 0.0, "high": 0.1},
    "focal_gamma": {"type": "float", "low": 0.0, "high": 3.0},
    "hidden_dims": {"type": "categorical", "choices": ["512-256", "1024-256", "1024-512-256"]},
}
OBJECTIVE = {"simulation": "binary", "coverages": [0.1, 0.3, 0.5], "blasts": [1.0, 0.5, 0.3]}
TARGET_ACCURACY = 0.98
THRESHOLD = 0.90
REPORTED = [("binary", 0.3, 1.0), ("binary", 0.3, 0.3)]   # also kept per trial: undiluted and 30% blasts at 30%


def objective_conditions():
    return [condition_name(OBJECTIVE["simulation"], c, 0.0, b)
            for b in OBJECTIVE["blasts"] for c in OBJECTIVE["coverages"]]


def reported_conditions():
    return [condition_name(s, c, 0.0, b) for s, c, b in REPORTED]


def parse_args():
    p = argparse.ArgumentParser(description="SPARSH: bounded hyperparameter study (Optuna)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--study_dir", default=None, help="Folder of the study (created by the first job)")
    p.add_argument("--base_run", default=None, help="Finished scripts/train.py run whose recipe the trials start from")
    p.add_argument("--max_trials", type=int, default=40, help="Stop once the study has this many finished or pruned trials")
    p.add_argument("--timeout_hours", type=float, default=None, help="Start no new trial after this long")
    p.add_argument("--startup_trials", type=int, default=5, help="Pruning starts after this many finished trials")
    p.add_argument("--warmup_epochs", type=int, default=40, help="No pruning before this epoch of fold 1")
    p.add_argument("--no_pruning", action="store_true")
    p.add_argument("--sampler_seed", type=int, default=None, help="TPE seed (default: random, so parallel jobs differ)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--report", action="store_true", help="Summarize the study; train nothing")
    p.add_argument("--score_runs", nargs="+", default=None, help="Score finished training runs; train nothing")
    return p.parse_args()


def setup_logging(log_file=None):
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handlers = [logging.StreamHandler()] + ([logging.FileHandler(log_file)] if log_file else [])
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def import_optuna():
    try:
        import optuna
    except ImportError:
        sys.exit("Optuna is not installed in this environment: pip install optuna")
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


def journal_storage(optuna, path: Path):
    try:
        from optuna.storages.journal import JournalFileBackend  # Optuna 4 and later
        backend = JournalFileBackend(str(path))
    except ImportError:  # Optuna 3.x
        backend = optuna.storages.JournalFileStorage(str(path))
    return optuna.storages.JournalStorage(backend)


# ------------------------------------------------------------------------------------------ scoring

def score_logits(y, logits, temperature_of, n_classes, idx_to_class):
    """Per-condition metrics and per-class recall, as scripts/train.py computes them."""
    rows, recalls = [], {}
    for name, lg in logits.items():
        probs = softmax_np(lg / temperature_of[:, None])
        m = summarize_probs(y, probs, THRESHOLD)
        conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
        m[f"callable_at_{TARGET_ACCURACY:.2f}"] = callable_at_accuracy(conf, correct, TARGET_ACCURACY)[0]
        rows.append({"condition": name, **m})
        pred = probs.argmax(axis=1)
        recalls[name] = [float(np.mean(pred[y == c] == c)) if (y == c).any() else np.nan for c in range(n_classes)]
    metrics = pd.DataFrame(rows)
    recall = pd.DataFrame(recalls, index=[idx_to_class[i] for i in range(n_classes)])
    recall.index.name = "class"
    return metrics, recall


def objective_value(metrics: pd.DataFrame) -> float:
    m = metrics.set_index("condition")
    missing = [c for c in objective_conditions() if c not in m.index]
    if missing:
        raise ValueError(f"objective conditions not scored: {missing}")
    return float(m.loc[objective_conditions(), "balanced_accuracy"].mean())


def score_run(run: Path) -> dict:
    """The objective and the reported conditions of a finished training run, from its saved predictions."""
    out = {"run": run.name}
    per = {}
    for name in dict.fromkeys(objective_conditions() + reported_conditions()):
        f = run / f"cv_predictions_{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: the run was not scored at {name}")
        p = pd.read_csv(f, usecols=["true_label", "prediction", "confidence"], dtype={"true_label": str,
                                                                                      "prediction": str})
        correct = (p["true_label"] == p["prediction"]).to_numpy()
        recall = p.assign(ok=correct).groupby("true_label")["ok"].mean()
        per[name] = {"balanced_accuracy": float(recall.mean()),
                     "callable": callable_at_accuracy(p["confidence"].to_numpy(), correct, TARGET_ACCURACY)[0]}
    out["objective"] = float(np.mean([per[c]["balanced_accuracy"] for c in objective_conditions()]))
    for name in reported_conditions():
        out[f"balanced_accuracy_{name}"] = per[name]["balanced_accuracy"]
        out[f"callable_at_{TARGET_ACCURACY:.2f}_{name}"] = per[name]["callable"]
    return out


def score_runs(paths):
    rows = []
    for p in paths:
        run = Path(p).expanduser()
        try:
            rows.append(score_run(run))
        except FileNotFoundError as e:
            sys.exit(str(e))
    df = pd.DataFrame(rows)
    df.insert(1, "group", df["run"].str.replace(r"_s\d+$", "", regex=True))
    pd.set_option("display.width", 220)
    print("Objective: mean recall over classes, averaged over "
          + ", ".join(objective_conditions()) + f"; callable share at {TARGET_ACCURACY:.0%} accuracy of calls")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    groups = df.groupby("group")
    if (groups.size() > 1).any():
        mean = groups.mean(numeric_only=True)
        mean.insert(0, "runs", groups.size())
        print("\nMean over the runs of each group (folder name without its final _s<seed>):")
        print(mean.to_string(float_format=lambda v: f"{v:.4f}"))


# ------------------------------------------------------------------------------------------ the study

def load_base(base_run: Path):
    """Data and recipe of a finished run, checked against the class counts it recorded."""
    cfg_path = base_run / "config.json"
    if not cfg_path.exists():
        sys.exit(f"No config.json in {base_run}: give a finished scripts/train.py run")
    config = json.loads(cfg_path.read_text())
    data = config["data"]
    label_map = load_label_map(data.get("label_map"))
    bundle = load_training_data(data["data_path"], label_map, data.get("exclude_ids"), data.get("cpg_list"),
                                data.get("group_col"), data.get("groups_file"))
    keep = filter_classes(bundle["labels"], data.get("min_samples", 5), data.get("exclude_classes") or [],
                          data.get("exclude_prefixes") or [])
    bundle = subset(bundle, keep)
    y, idx_to_class = encode_labels(bundle["labels"])
    counts = {idx_to_class[i]: int(n) for i, n in enumerate(np.bincount(y, minlength=len(idx_to_class)))}
    if counts != data.get("class_counts"):
        recorded = data.get("class_counts") or {}
        diff = {k: (recorded.get(k), counts.get(k)) for k in sorted(set(recorded) | set(counts))
                if recorded.get(k) != counts.get(k)}
        sys.exit(f"The training data no longer match {base_run.name} (class: recorded, now): {diff}. "
                 "The study would tune on different data; retrain the base run or fix the files.")
    names = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(config["training"]) - names)
    if unknown:
        logger.warning(f"Settings of the base run unknown to this version, ignored: {unknown}")
    cfg = TrainConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in config["training"].items() if k in names})
    return config, cfg, dict(config["model"]), bundle, y, idx_to_class


def base_params(cfg: TrainConfig, model_config: dict) -> dict:
    return {"learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
            "dropout": float(model_config["dropout"]), "label_smoothing": cfg.label_smoothing,
            "focal_gamma": cfg.focal_gamma,
            "hidden_dims": "-".join(str(int(v)) for v in model_config["hidden_dims"])}


def in_space(params: dict) -> list:
    outside = []
    for k, s in SPACE.items():
        v = params[k]
        if s["type"] == "categorical":
            ok = v in s["choices"]
        else:
            ok = s["low"] - 1e-12 <= v <= s["high"] + 1e-12
        if not ok:
            outside.append(f"{k}={v}")
    return outside


def suggest(trial) -> dict:
    p = {}
    for name, s in SPACE.items():
        if s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        else:
            p[name] = trial.suggest_categorical(name, s["choices"])
    return p


def as_train_args(params: dict) -> str:
    return (f"--learning_rate {params['learning_rate']:.3g} --weight_decay {params['weight_decay']:.3g} "
            f"--dropout {params['dropout']:.3f} --label_smoothing {params['label_smoothing']:.3f} "
            f"--focal_gamma {params['focal_gamma']:.2f} --hidden_dims {params['hidden_dims'].replace('-', ' ')}")


def check_space_file(study_dir: Path, base_run: Path, config: dict):
    spec = {"space": SPACE, "objective": OBJECTIVE, "base_run": str(base_run.resolve()),
            "base_run_id": config.get("run_id")}
    f = study_dir / "space.json"
    if f.exists():
        saved = json.loads(f.read_text())
        for key in ("space", "objective", "base_run", "base_run_id"):
            if saved.get(key) != json.loads(json.dumps(spec[key])):
                sys.exit(f"{f}: the {key} of this study does not match this job ({saved.get(key)} against "
                         f"{spec[key]}). Use a new --study_dir for a different study.")
    else:
        f.write_text(json.dumps({**spec, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


def run_study(args):
    optuna = import_optuna()
    study_dir = Path(args.study_dir).expanduser()
    base_run = Path(args.base_run).expanduser()
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)
    job = os.environ.get("PBS_JOBID", str(os.getpid())).split(".")[0]
    setup_logging(study_dir / f"tune_log_{job}.txt")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA was requested but no GPU is visible. Submit to a GPU node, or pass --device cpu deliberately.")

    config, cfg, model_config, bundle, y, idx_to_class = load_base(base_run)
    check_space_file(study_dir, base_run, config)
    X, sample_ids, groups = bundle["X"], bundle["sample_ids"], bundle["groups"]
    n_classes = len(idx_to_class)
    normal_idx = normal_index(idx_to_class, cfg.normal_class)
    if normal_idx is None:
        sys.exit(f"The objective scores diluted samples, which needs the class {cfg.normal_class!r}")
    logger.info(f"Base run {base_run.name}: {len(y)} samples, {X.shape[1]} CpGs, {n_classes} classes, seed {cfg.seed}")
    logger.info(f"Objective conditions: {objective_conditions()}")

    storage = journal_storage(optuna, study_dir / "study.journal")
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True, constant_liar=True),
        pruner=(optuna.pruners.NopPruner() if args.no_pruning else
                optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials, n_warmup_steps=args.warmup_epochs)))
    base = base_params(cfg, model_config)
    outside = in_space(base)
    if outside:
        logger.warning(f"The base run's settings lie outside the search space ({outside}); no baseline trial")
    elif not study.trials:
        study.enqueue_trial(base, user_attrs={"baseline": True}, skip_if_exists=True)
        logger.info(f"Baseline trial queued: {base}")

    def objective(trial):
        params = suggest(trial)
        cfg_t = replace(cfg, learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                        label_smoothing=params["label_smoothing"], focal_gamma=params["focal_gamma"],
                        eval_sims=(OBJECTIVE["simulation"],), eval_coverages=tuple(OBJECTIVE["coverages"]),
                        eval_blasts=tuple(OBJECTIVE["blasts"]), eval_call_errors=(0.0,), device=args.device)
        model_t = dict(model_config, dropout=params["dropout"],
                       hidden_dims=[int(v) for v in params["hidden_dims"].split("-")])
        tdir = study_dir / "trials" / f"trial_{trial.number:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "settings.json").write_text(json.dumps(
            {"params": params, "train_args": as_train_args(params), "model": model_t, "training": cfg_t.to_dict(),
             "baseline": bool(trial.user_attrs.get("baseline"))}, indent=2))
        logger.info(f"Trial {trial.number}: {params}")

        def report(tag, epoch, val_nll, val_acc):
            if tag != "fold1":
                return
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"fold 1, epoch {epoch}: inner-validation accuracy {val_acc:.3f}")

        set_deterministic_mode(cfg_t.seed)
        t0 = time.time()
        try:
            cv = cross_validate(X, y, sample_ids, groups, n_classes, model_t, cfg_t, None, None, normal_idx,
                                epoch_callback=None if args.no_pruning else report)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics, recall = score_logits(y, cv["logits"], cv["temperature_of"], n_classes, idx_to_class)
        value = objective_value(metrics)
        metrics.to_csv(tdir / "metrics.csv", index=False, float_format="%.4f")
        recall[objective_conditions()].to_csv(tdir / "recall_by_class.csv", float_format="%.3f")
        pd.DataFrame(cv["records"]).to_csv(tdir / "folds_summary.csv", index=False)
        m = metrics.set_index("condition")
        for name in reported_conditions():
            trial.set_user_attr(f"balanced_accuracy_{name}", float(m.loc[name, "balanced_accuracy"]))
            trial.set_user_attr(f"callable_at_{TARGET_ACCURACY:.2f}_{name}",
                                float(m.loc[name, f"callable_at_{TARGET_ACCURACY:.2f}"]))
        trial.set_user_attr("minutes", round((time.time() - t0) / 60.0, 1))
        trial.set_user_attr("best_epochs", [int(r["best_epoch"]) for r in cv["records"]])
        logger.info(f"Trial {trial.number}: objective {value:.4f} ({(time.time() - t0) / 60:.0f} min)")
        return value

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done_before = sum(t.state in states for t in study.trials)
    if done_before >= args.max_trials:
        logger.info(f"The study already has {done_before} finished or pruned trials (--max_trials {args.max_trials})")
        return
    study.optimize(objective, timeout=None if args.timeout_hours is None else args.timeout_hours * 3600,
                   catch=(RuntimeError,), gc_after_trial=True,
                   callbacks=[optuna.study.MaxTrialsCallback(args.max_trials, states=states)])
    done = sum(t.state in states for t in study.get_trials(deepcopy=False))
    logger.info(f"The study has {done} finished or pruned trials; "
                f"report with: python scripts/tune.py --study_dir {study_dir} --report")


def report(args):
    optuna = import_optuna()
    study_dir = Path(args.study_dir).expanduser()
    if not (study_dir / "study.journal").exists():
        sys.exit(f"No study in {study_dir}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=journal_storage(optuna, study_dir / "study.journal"))
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df = df.sort_values("value", ascending=False, na_position="last")
    df.to_csv(study_dir / "trials.csv", index=False)
    print(f"{len(df)} trials: " + ", ".join(f"{k.lower()} {v}" for k, v in df["state"].value_counts().items()))
    done = df[df["state"] == "COMPLETE"]
    if done.empty:
        print("No finished trial yet.")
        return
    cols = ["number", "value"] + [c for c in df.columns if c.startswith("params_")] + \
           [c for c in df.columns if c.startswith("user_attrs_balanced") or c.startswith("user_attrs_callable")]
    shown = done[cols].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    pd.set_option("display.width", 250)
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    base = df[df.get("user_attrs_baseline", pd.Series(False, index=df.index)).fillna(False).astype(bool)]
    if len(base) and base["state"].iloc[0] == "COMPLETE":
        print(f"\nBaseline (the base run's settings), trial {int(base['number'].iloc[0])}: "
              f"objective {base['value'].iloc[0]:.4f}")
    print("\nThe two best settings, as scripts/train.py options (EXTRA_ARGS of jobs/train.pbs):")
    for _, r in done.head(2).iterrows():
        params = {k.replace("params_", ""): r[k] for k in r.index if k.startswith("params_")}
        print(f"  trial {int(r['number'])} ({r['value']:.4f}): {as_train_args(params)}")
    print(f"\nWritten {study_dir / 'trials.csv'}")


def main():
    args = parse_args()
    if args.score_runs:
        score_runs(args.score_runs)
    elif args.report:
        if not args.study_dir:
            sys.exit("--report needs --study_dir")
        report(args)
    else:
        if not (args.study_dir and args.base_run):
            sys.exit("A study needs --study_dir and --base_run (or use --report or --score_runs)")
        run_study(args)


if __name__ == "__main__":
    main()
