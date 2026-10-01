#!/usr/bin/env python3
r"""
SPARSH for one sample, from its nanopore reads to its report, in one command.

    python scripts/run_sample.py --sample S01 --bam /data/run/bam_pass/barcode01 --out_dir results/S01 \
        --reference hg19_all.fasta --manifest EPIC-8v2-0_A2.csv --model_dir /models/v04_labels \
        --panel cn_panel.tsv.gz --sex F --blast_percent 80

--bam takes one or more BAM files, or folders of BAM files such as a barcode folder of a run's bam_pass, with the
reads aligned to the reference or not aligned at all; in a folder, files of failed reads (_fail_ in the name) are
left out. The work is done in steps, each in its own folder of --out_dir:

  bam/           the reads as one sorted and indexed file, <sample>.bam. Files aligned to --reference (the same
                 sequence names and lengths as its .fai index) are merged and sorted; a single such file that is
                 already sorted is linked, not copied. Otherwise the reads are aligned again: files aligned to another
                 reference are first returned to unaligned reads (samtools reset, primary records only), and all are
                 merged and aligned with dorado aligner, as the laboratory's pipeline does, keeping the tags of the
                 modified-base calls (--aligner minimap2 aligns with samtools fastq and minimap2 -y instead). The
                 number of primary records is checked against the input.
  methylation/   modkit pileup --modified-bases 5mC 5hmC --combine-mods --cpg, the command of the laboratory's
                 MethPipe pipeline, and scripts/bedmethyl_to_csv.py, which adds the calls of the two strands of each
                 probe's CpG (--strands forward takes the forward strand alone, as MethPipe does): <sample>.csv, the
                 input of the model, and the pileup, compressed.
  prediction/    scripts/predict.py with --model_dir.
  copy_number/   reads counted in 100 kb bins (primary alignments with mapping quality 20 or more) and the copy-number
                 call against the stored reference --panel (made with scripts/karyotype.py panel from a counted
                 cohort sequenced the same way); --no_panel calls with GC correction only, for a first look.
  report/        scripts/report.py; the page is also copied to <out_dir>/<sample>.report.html.

--sex (F or M) and --blast_percent go into sample_sheet.csv, which the copy-number call (the sex, and the leukaemic
fraction within 20 points of the blast percentage) and the report use. A finished step is not run again, so a
stopped run continues with the same command; --redo STEP runs that step and the ones after it again. run.log holds
every command and its output; run_info.json the inputs, settings, tool versions and the time of each step.

Tools are taken from the options, else from the environment variables SAMTOOLS, MODKIT, DORADO and MINIMAP2, else
from PATH. MethPipe runs modkit 0.6.1 and dorado 1.3.0; other versions are noted in run_info.json and run.log.
"""

import argparse
import csv
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEPS = ("bam", "methylation", "prediction", "copy_number", "report")
MODKIT_ARGS = ["--modified-bases", "5mC", "5hmC", "--combine-mods", "--cpg"]
METHPIPE_VERSIONS = {"modkit": "0.6.1", "dorado": "1.3.0"}


class StepError(RuntimeError):
    pass


class Runner:
    """Runs commands, writing each command and its output to run.log."""

    def __init__(self, log_path: Path):
        self.log = open(log_path, "a")

    def note(self, text: str) -> None:
        self.log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        self.log.flush()

    def run(self, cmd, shell: bool = False, stdout=None) -> None:
        shown = cmd if shell else " ".join(shlex.quote(str(c)) for c in cmd)
        self.log.write(f"$ {shown}\n")
        self.log.flush()
        if shell:
            args = ["bash", "-o", "pipefail", "-c", cmd]
        else:
            args = [str(c) for c in cmd]
        with (open(stdout, "wb") if stdout else open(os.devnull, "wb")) as out:
            result = subprocess.run(args, stdout=out if stdout else self.log, stderr=self.log)
        self.log.flush()
        if result.returncode != 0:
            raise StepError(f"command failed (status {result.returncode}): {shown[:200]}; see run.log")

    def capture(self, cmd) -> str:
        result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            raise StepError(f"{cmd[0]} {cmd[1]} failed: {result.stderr.strip()[:200]}")
        return result.stdout


def tool(option, variable, name):
    """The tool's path from the option, else the environment variable, else PATH; None when there is none."""
    for given, origin in ((option, f"--{name}"), (os.environ.get(variable), variable)):
        if given:
            found = shutil.which(given)
            if not found:
                sys.exit(f"{origin}: {given} is not an executable file")
            return found
    return shutil.which(name)


def version(path, name):
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout + out.stderr).strip()
    # a log line may come first; its date and time are not the version
    clean = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?", " ", text)
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", clean)
    return m.group(1) if m else (text.splitlines()[0] if text else None)


# ---------------------------------------------------------------------------------------------- inputs
def expand_inputs(paths):
    """The BAM files given, with the files ending in .bam directly inside each folder given; in a folder, files of
    failed reads (MinKNOW puts _fail_ in their names) and hidden files are left out."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(f for f in p.glob("*.bam") if not f.name.startswith(".") and "_fail_" not in f.name)
            if not found:
                raise SystemExit(f"--bam: no BAM files of passed reads in {p}")
            files += found
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"--bam: {p} does not exist")
    unique = list(dict.fromkeys(f.resolve() for f in files))
    return unique


def reference_sequences(fasta: Path):
    fai = Path(str(fasta) + ".fai")
    if not fai.exists():
        raise SystemExit(f"{fai} not found: index the reference with samtools faidx")
    seqs = {}
    for line in fai.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            seqs[f[0]] = int(f[1])
    return seqs


def bam_state(runner, samtools, bam, reference):
    """('aligned' | 'unaligned' | 'other', coordinate-sorted) from the header."""
    header = runner.capture([samtools, "view", "-H", bam])
    seqs, sorted_ = {}, False
    for line in header.splitlines():
        if line.startswith("@SQ"):
            tags = dict(t.split(":", 1) for t in line.split("\t")[1:] if ":" in t)
            if "SN" in tags and tags.get("LN", "").isdigit():
                seqs[tags["SN"]] = int(tags["LN"])
        elif line.startswith("@HD"):
            sorted_ = "SO:coordinate" in line
    if not seqs:
        return "unaligned", False
    return ("aligned" if seqs == reference else "other"), sorted_


def count_records(runner, samtools, bam, primary_only, threads="2"):
    flags = ["-F", "0x900"] if primary_only else []
    return int(runner.capture([samtools, "view", "-c", "-@", threads] + flags + [bam]).strip())


# ---------------------------------------------------------------------------------------------- steps
def step_bam(cfg, runner, out_dir):
    s, sam, threads = cfg["sample"], cfg["samtools"], str(cfg["threads"])
    folder = out_dir / "bam"
    tmp = folder / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    final = folder / f"{s}.bam"
    reference = reference_sequences(Path(cfg["reference"]))
    states = [bam_state(runner, sam, b, reference) for b in cfg["inputs"]]
    kinds = [k for k, _ in states]
    info = {"inputs": len(cfg["inputs"]), "aligned": kinds.count("aligned"), "unaligned": kinds.count("unaligned"),
            "other_reference": kinds.count("other")}
    if all(k == "aligned" for k in kinds):
        linked = len(cfg["inputs"]) == 1 and states[0][1]
        total = None if linked else sum(count_records(runner, sam, b, False, threads) for b in cfg["inputs"])
        if linked:
            final.symlink_to(cfg["inputs"][0])
            info["action"] = "linked the sorted input"
            source = Path(cfg["inputs"][0])
            for index in (Path(str(source) + ".bai"), source.with_suffix(".bai"), Path(str(source) + ".csi")):
                if index.exists() and index.stat().st_mtime >= source.stat().st_mtime:
                    Path(str(final) + index.suffix).symlink_to(index)
                    break
        elif all(sorted_ for _, sorted_ in states):
            runner.run([sam, "merge", "-f", "-@", threads, "-o", final] + cfg["inputs"])
            info["action"] = "merged sorted inputs"
        else:
            source = cfg["inputs"][0]
            if len(cfg["inputs"]) > 1:
                source = tmp / "merged.bam"
                runner.run([sam, "merge", "-f", "-@", threads, "-o", source] + cfg["inputs"])
            runner.run([sam, "sort", "-@", threads, "-m", "1G", "-T", tmp / "sort", "-o", final, source])
            info["action"] = "merged and sorted" if len(cfg["inputs"]) > 1 else "sorted"
        if not any(Path(str(final) + suffix).exists() for suffix in (".bai", ".csi")):
            runner.run([sam, "index", "-@", threads, final])
        if total is not None:
            got = count_records(runner, sam, final, False, threads)
            if got != total:
                raise StepError(f"{final.name} holds {got} records, the inputs {total}")
    else:
        parts = []
        for i, (b, kind) in enumerate(zip(cfg["inputs"], kinds)):
            if kind == "unaligned":
                parts.append(b)
            else:                       # aligned elsewhere: back to unaligned reads in their own orientation
                part = tmp / f"reset_{i}.bam"
                runner.run(f"{shlex.quote(sam)} view -u -F 0x900 {shlex.quote(str(b))} | "
                           f"{shlex.quote(sam)} reset -o {shlex.quote(str(part))} -", shell=True)
                parts.append(part)
        unaligned = parts[0]
        if len(parts) > 1:
            unaligned = tmp / "unaligned.bam"
            runner.run([sam, "merge", "-f", "-@", threads, "-o", unaligned] + parts)
        expected = count_records(runner, sam, unaligned, True, threads)
        aligned = tmp / "aligned.bam"
        if cfg["aligner"] == "dorado":
            runner.run([cfg["dorado"], "aligner", "-t", threads, cfg["reference"], unaligned], stdout=aligned)
        else:
            index = Path(cfg["reference"] + ".map-ont.mmi")
            target = index if index.exists() else Path(cfg["reference"])
            runner.run(f"{shlex.quote(sam)} fastq -@ {threads} -T MM,ML,MN {shlex.quote(str(unaligned))} | "
                       f"{shlex.quote(cfg['minimap2'])} -t {threads} -ax map-ont -y {shlex.quote(str(target))} - | "
                       f"{shlex.quote(sam)} view -b -o {shlex.quote(str(aligned))} -", shell=True)
        runner.run([sam, "sort", "-@", threads, "-m", "1G", "-T", tmp / "sort", "-o", final, aligned])
        runner.run([sam, "index", "-@", threads, final])
        got = count_records(runner, sam, final, True, threads)
        if got != expected:
            raise StepError(f"{final.name} holds {got} primary records, the reads {expected}")
        info["action"] = f"aligned with {cfg['aligner']}"
    shutil.rmtree(tmp, ignore_errors=True)
    return info, f"{info['inputs']} file(s): {info['action']}"


def step_methylation(cfg, runner, out_dir):
    s = cfg["sample"]
    folder = out_dir / "methylation"
    folder.mkdir(parents=True, exist_ok=True)
    bed = folder / f"{s}_pileup.bed"
    runner.run([cfg["modkit"], "pileup", "--ref", cfg["reference"], "-t", str(cfg["threads"])] + MODKIT_ARGS
               + [out_dir / "bam" / f"{s}.bam", bed])
    runner.run([sys.executable, REPO / "scripts" / "bedmethyl_to_csv.py", "--bedmethyl", bed, "--manifest",
                cfg["manifest"], "--sample", s, "--out", folder / f"{s}.csv", "--strands", cfg["strands"]])
    with open(bed, "rb") as src, gzip.open(str(bed) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    bed.unlink()
    with open(folder / f"{s}.csv") as fh:
        probes = len(fh.readline().rstrip("\n").split(",")) - 1
    return {"probes_with_a_call": probes}, f"{probes:,} probes with a call"


def step_prediction(cfg, runner, out_dir):
    s = cfg["sample"]
    runner.run([sys.executable, REPO / "scripts" / "predict.py", "--model_dir", cfg["model_dir"], "--samples",
                out_dir / "methylation" / f"{s}.csv", "--output_dir", out_dir / "prediction", "--device",
                cfg["device"]])
    with open(out_dir / "prediction" / "predictions.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    return {"coverage_pct": float(row["coverage_pct"])}, f"coverage of the model's CpGs {float(row['coverage_pct']):.1f}%"


def step_copy_number(cfg, runner, out_dir):
    s, sam = cfg["sample"], cfg["samtools"]
    folder = out_dir / "copy_number"
    bam = out_dir / "bam" / f"{s}.bam"
    runner.run(f"{shlex.quote(sam)} view -@ 2 -F 0x904 -q 20 {shlex.quote(str(bam))} | cut -f 2-6 | "
               f"{shlex.quote(sys.executable)} {shlex.quote(str(REPO / 'scripts' / 'karyotype.py'))} count "
               f"--sample {shlex.quote(s)} --build hg19 --out_dir {shlex.quote(str(folder / 'counts'))} "
               f"--source {shlex.quote(str(bam))}", shell=True)
    cmd = [sys.executable, REPO / "scripts" / "karyotype.py", "call", "--counts_dir", folder / "counts",
           "--out_dir", folder / "calls", "--build", "hg19", "--samples", s]
    cmd += ["--panel", cfg["panel"]] if cfg["panel"] else ["--no_reference"]
    if cfg["sheet"]:
        cmd += ["--sample_sheet", cfg["sheet"]]
    runner.run(cmd)
    calls = list((folder / "calls").glob("*.karyotype.json"))
    if not calls:
        return {"called": False}, "not called (see copy_number/calls/karyotype_summary.csv)"
    result = json.loads(calls[0].read_text())
    return ({"called": True, "qc_status": result["qc"]["status"], "bin_size": result["qc"]["bin_size"]},
            f"called at bins of {result['qc']['bin_size']:,} bases, quality {result['qc']['status']}")


def step_report(cfg, runner, out_dir):
    s = cfg["sample"]
    cmd = [sys.executable, REPO / "scripts" / "report.py", "--predictions_dir", out_dir / "prediction",
           "--output_dir", out_dir / "report"]
    if (out_dir / "copy_number" / "calls").is_dir():
        cmd += ["--karyotype_dir", out_dir / "copy_number" / "calls"]
    if cfg["sheet"]:
        cmd += ["--sample_sheet", cfg["sheet"]]
    runner.run(cmd)
    pages = sorted((out_dir / "report" / "cases").glob("*.html"))
    if len(pages) != 1:
        raise StepError(f"expected one report page, found {len(pages)}")
    shutil.copy2(pages[0], out_dir / f"{s}.report.html")
    return {"page": f"{s}.report.html"}, f"{s}.report.html"


STEP_FUNCTIONS = {"bam": step_bam, "methylation": step_methylation, "prediction": step_prediction,
                  "copy_number": step_copy_number, "report": step_report}


# ---------------------------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, help="sample name: letters, digits, '.', '_' and '-'")
    p.add_argument("--bam", nargs="+", required=True, help="BAM files or folders of BAM files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--reference", required=True, help="GRCh37 FASTA with its .fai (hg19_all.fasta)")
    p.add_argument("--manifest", required=True, help="Illumina's MethylationEPIC v2.0 manifest (CSV)")
    p.add_argument("--model_dir", required=True, help="the model folder (for example v04_labels)")
    p.add_argument("--panel", default=None, help="stored copy-number reference (scripts/karyotype.py panel)")
    p.add_argument("--no_panel", action="store_true", help="call copy number without a reference, for a first look")
    p.add_argument("--sex", choices=["F", "M"], default=None)
    p.add_argument("--blast_percent", type=float, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--aligner", choices=["dorado", "minimap2"], default="dorado",
                   help="for reads that need aligning (default dorado aligner, as the laboratory's pipeline)")
    p.add_argument("--samtools", default=None)
    p.add_argument("--modkit", default=None)
    p.add_argument("--dorado", default=None)
    p.add_argument("--minimap2", default=None)
    p.add_argument("--strands", choices=["both", "forward"], default="both",
                   help="the strands of each CpG behind the model's input: both (default), or forward as MethPipe")
    p.add_argument("--device", default="cpu", help="for the prediction (default cpu)")
    p.add_argument("--redo", choices=STEPS, default=None, help="run this step and the ones after it again")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.sample):
        sys.exit("--sample may hold only letters, digits, '.', '_' and '-'")
    if bool(args.panel) == bool(args.no_panel):
        sys.exit("give either --panel (a stored copy-number reference) or --no_panel")
    if args.blast_percent is not None and not 0 < args.blast_percent <= 100:
        sys.exit("--blast_percent must be above 0 and at most 100")
    for label, path in (("--reference", args.reference), ("--manifest", args.manifest),
                        ("--model_dir", args.model_dir), ("--panel", args.panel)):
        if path and not Path(path).exists():
            sys.exit(f"{label}: {path} does not exist")
    sys.path.insert(0, str(REPO / "scripts"))
    from bedmethyl_to_csv import check_manifest
    check_manifest(Path(args.manifest))                     # before any step, not after hours of alignment
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = {"samtools": tool(args.samtools, "SAMTOOLS", "samtools"), "modkit": tool(args.modkit, "MODKIT", "modkit"),
             "dorado": tool(args.dorado, "DORADO", "dorado"), "minimap2": tool(args.minimap2, "MINIMAP2", "minimap2")}
    for name in ("samtools", "modkit"):
        if not tools[name]:
            sys.exit(f"{name} not found: give --{name} or put it on PATH")
    versions = {name: version(path, name) for name, path in tools.items()}

    sheet = None
    if args.sex or args.blast_percent is not None:
        sheet = out_dir / "sample_sheet.csv"
        blast = "" if args.blast_percent is None else f"{args.blast_percent:g}"
        sheet.write_text(f"sample,sex,blast_percent\n{args.sample},{args.sex or ''},{blast}\n")
    cfg = {"sample": args.sample, "inputs": [str(p) for p in expand_inputs(args.bam)],
           "reference": str(Path(args.reference).resolve()), "manifest": str(Path(args.manifest).resolve()),
           "model_dir": str(Path(args.model_dir).resolve()), "panel": str(Path(args.panel).resolve()) if args.panel
           else None, "sheet": str(sheet) if sheet else None, "threads": args.threads, "aligner": args.aligner,
           "device": args.device, "strands": args.strands, **tools}

    info_path = out_dir / "run_info.json"
    earlier = json.loads(info_path.read_text()).get("settings", {}) if info_path.exists() else {}
    if (out_dir / ".methylation.done").exists() and args.redo not in ("bam", "methylation"):
        made_with = earlier.get("strands", "forward")          # runs made before the option used the forward strand
        if made_with != args.strands:
            sys.exit(f"The methylation step of this run was made with --strands {made_with}; give --redo methylation "
                     f"to make it again with --strands {args.strands}")

    if args.redo:
        for step in STEPS[STEPS.index(args.redo):]:
            shutil.rmtree(out_dir / step, ignore_errors=True)
            (out_dir / f".{step}.done").unlink(missing_ok=True)
        (out_dir / f"{args.sample}.report.html").unlink(missing_ok=True)
    info = json.loads(info_path.read_text()) if info_path.exists() else {"steps": {}}
    info.update({"sample": args.sample, "inputs": cfg["inputs"], "settings": {k: cfg[k] for k in (
        "reference", "manifest", "model_dir", "panel", "threads", "aligner", "device", "strands")}, "tools": tools,
        "versions": versions, "sparsh_version": (REPO / "VERSION").read_text().strip()})
    runner = Runner(out_dir / "run.log")
    runner.note(f"run_sample.py {' '.join(shlex.quote(a) for a in sys.argv[1:])}")
    for name, wanted in METHPIPE_VERSIONS.items():
        got = versions.get(name)
        if tools.get(name) and got and got != wanted:
            runner.note(f"{name} {got}; MethPipe runs {wanted}")
            print(f"Note: {name} {got}; MethPipe runs {wanted}")

    print(f"{args.sample}: {len(cfg['inputs'])} BAM file(s) -> {out_dir}")
    for step in STEPS:
        marker = out_dir / f".{step}.done"
        if marker.exists():
            print(f"  {step:<12} done before")
            continue
        shutil.rmtree(out_dir / step, ignore_errors=True)
        (out_dir / step).mkdir(parents=True)
        runner.note(f"step {step}")
        started = time.time()
        try:
            if step == "bam":
                needs = any(bam_state(runner, cfg["samtools"], b, reference_sequences(Path(cfg["reference"])))[0]
                            != "aligned" for b in cfg["inputs"])
                if needs and not cfg.get(cfg["aligner"]):
                    raise StepError(f"the reads need aligning and {cfg['aligner']} was not found: give "
                                    f"--{cfg['aligner']} or put it on PATH")
            details, text = STEP_FUNCTIONS[step](cfg, runner, out_dir)
        except StepError as exc:
            info["steps"][step] = {"status": "failed", "error": str(exc)}
            info_path.write_text(json.dumps(info, indent=2))
            print(f"  {step:<12} FAILED: {exc}")
            return 1
        minutes = round((time.time() - started) / 60, 1)
        info["steps"][step] = {"status": "done", "minutes": minutes, **details}
        info_path.write_text(json.dumps(info, indent=2))
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
        print(f"  {step:<12} {text} ({minutes} min)")
    print(f"Report: {out_dir / (args.sample + '.report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
