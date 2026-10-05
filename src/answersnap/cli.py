"""Command-line entry point.

Exit codes: 0 done, 1 bad config or usage (nothing was spent), 2 the run is
incomplete — an engine was skipped or calls failed. A report is still written
for exit 2, and it says what is missing.
"""

import argparse
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from answersnap import __version__, report, run, store
from answersnap.config import (
    MAX_REPEATS,
    SUPPORTED_ENGINES,
    ConfigError,
    load_config,
    parse_config,
    prompt_identity,
)
from answersnap.engines import build_engines, key_names, key_present, label, live_status
from answersnap.providers import get_provider_class, registered_platforms

# ---------------------------------------------------------------- constants
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INCOMPLETE = 2
DEFAULT_CONFIG_NAME = "answersnap.yaml"
DEFAULT_OUT = Path("output") / "answersnap"


def _now():
    return datetime.now(timezone.utc)


def _say(message=""):
    print(message, flush=True)


# ---------------------------------------------------------------- init
def cmd_init(args):
    target = Path(args.out)
    if target.exists():
        _say(f"{target} already exists; not overwriting it.")
        return EXIT_USAGE
    example = resources.files("answersnap.examples").joinpath("example.yaml")
    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    _say(f"Wrote {target}. Try it without spending anything:\n"
         f"  answersnap run --config {target} --dry-run")
    return EXIT_OK


# ---------------------------------------------------------------- run
def _selected_engines(config, requested):
    if not requested:
        return list(config.engines)
    names = [n.strip() for n in requested.split(",") if n.strip()]
    unknown = [n for n in names if n not in config.engines]
    if unknown:
        raise ConfigError(f"--engines {', '.join(unknown)} not in the config's engines "
                          f"({', '.join(config.engines)})")
    return names


def _print_plan(tasks, cost, skipped, dry_run):
    _say(f"Plan: {len(tasks)} calls" + (" (dry run: fixtures, no API calls)" if dry_run else ""))
    for engine, (calls, usd) in cost.items():
        price = "no cost" if dry_run else ("no estimate" if usd is None else f"≈ ${usd:.2f}")
        _say(f"  {label(engine):8} {calls:4} calls  {price}")
    for _, reason in skipped.values():
        _say(f"  skipped: {reason}")


def _confirmed(args, dry_run):
    if dry_run or args.yes:
        return True
    if not sys.stdin.isatty():
        _say("Not running: confirm with --yes when not at a terminal.")
        return False
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def _prepare_new(args):
    config = load_config(args.config)
    for note in config.warnings():
        _say(f"note: {note}")
    if not args.dry_run:
        store.refuse_if_prompt_set_changed(args.out, config.brand.name,
                                           config.prompt_set.version,
                                           prompt_identity(config))
    run_dir, run_id = store.new_run_dir(args.out, config.brand.name)
    repeats = args.repeats if args.repeats is not None else config.repeats
    return config, run_dir, run_id, frozenset(), _now(), repeats


def _prepare_resume(args):
    if args.repeats is not None:
        raise ConfigError("--repeats cannot be combined with --resume: a run keeps the "
                          "repeats it started with")
    run_dir = Path(args.resume)
    earlier = store.read_manifest(run_dir)
    requested = "dry_run" if args.dry_run else "live"
    if earlier["mode"] != requested:
        # Mixing fixture answers into a live run (or the reverse) would put
        # invented answers next to real ones under one label.
        raise ConfigError(f"{run_dir} is a {earlier['mode'].replace('_', ' ')} run; resume it "
                          f"{'with' if earlier['mode'] == 'dry_run' else 'without'} --dry-run")
    config = parse_config(store.read_frozen_config(run_dir), source=str(run_dir))
    started = datetime.fromisoformat(earlier["started_at"])
    return (config, run_dir, earlier["run_id"], store.done_pairs(run_dir), started,
            earlier["repeats"])


def _repeats(value):
    number = int(value)
    if not 1 <= number <= MAX_REPEATS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_REPEATS}")
    return number


def cmd_run(args):
    prepare = _prepare_resume if args.resume else _prepare_new
    config, run_dir, run_id, done, started, repeats = prepare(args)
    selected = _selected_engines(config, args.engines)
    engines, skipped = build_engines(selected, dry_run=args.dry_run)

    tasks = run.plan_tasks(config, list(engines), repeats, done)
    capped = args.max_calls is not None and len(tasks) > args.max_calls
    if capped:
        tasks = tasks[:args.max_calls]
    cost = run.estimate_cost(tasks, {name: engine.model for name, engine in engines.items()})
    _print_plan(tasks, cost, skipped, args.dry_run)
    if capped:
        _say(f"  capped at --max-calls {args.max_calls}; the run will be marked incomplete")
    if not _confirmed(args, args.dry_run):
        return EXIT_USAGE

    manifest = run.initial_manifest(
        config, run_id=run_id, mode="dry_run" if args.dry_run else "live", engines=engines,
        skipped=skipped, tasks=tasks, repeats=repeats, max_calls=args.max_calls,
        started_at=started, cost=cost, keep_raw=not args.no_raw)
    store.write_frozen_config(run_dir, config.to_dict())
    store.write_manifest(run_dir, manifest)
    try:
        run.execute(config, run_dir, manifest, engines, tasks,
                    workers=1 if args.dry_run else args.workers)
    finally:
        run.finalise(manifest, run_dir, _now())
        store.write_manifest(run_dir, manifest)

    paths = report.generate(run_dir, fetch=not args.no_fetch)
    _say(f"\nRun {manifest['status'].replace('_', ' ')}: {run_dir}")
    for path in paths:
        _say(f"  {path}")
    return _exit_for(manifest["status"])


def _exit_for(status):
    return EXIT_OK if status in (run.STATUS_COMPLETE, run.STATUS_COMPLETE_SUBSET) else EXIT_INCOMPLETE


# ---------------------------------------------------------------- report
def cmd_report(args):
    paths = report.generate(args.run_dir, fetch=not args.no_fetch)
    for path in paths:
        _say(str(path))
    return _exit_for(store.read_manifest(args.run_dir)["status"])


# ---------------------------------------------------------------- verify-providers
def cmd_verify_providers(args):
    _say(f"{'platform':12} {'supported':10} verified")
    for platform in registered_platforms():
        verified = get_provider_class(platform).response_shape_verified()
        supported = "yes" if platform in SUPPORTED_ENGINES else "no"
        _say(f"{platform:12} {supported:10} {'yes' if verified else 'no'}")
    return EXIT_OK


# ---------------------------------------------------------------- doctor
def cmd_doctor(args):
    ready = 0
    for engine in SUPPORTED_ENGINES:
        status, reason = live_status(engine)
        ready += status == "ok"
        key = "set" if key_present(engine) else "not set"
        _say(f"{label(engine):8} {key_names(engine)} {key:8} " + ("ready" if status == "ok" else reason))
    healthy = ready > 0
    if not healthy:
        _say("no engine is ready for a live run (dry runs still work)")
    if args.config:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            _say(str(exc))
            return EXIT_USAGE
        _say(f"config ok: {config.brand.name}, {len(config.prompt_set.queries)} questions, "
             f"engines {', '.join(config.engines)}")
        for note in config.warnings():
            _say(f"note: {note}")
            healthy = False
    return EXIT_OK if healthy else EXIT_INCOMPLETE


# ---------------------------------------------------------------- parser
def build_parser():
    parser = argparse.ArgumentParser(
        prog="answersnap",
        description="Auditable snapshots of what AI answer engines say about a brand.")
    parser.add_argument("--version", action="version", version=f"answersnap {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", help="write an example config to edit")
    p.add_argument("--out", default=DEFAULT_CONFIG_NAME)
    p.set_defaults(handler=cmd_init)

    p = sub.add_parser("run", help="ask every question, store the answers, write the report")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="config file (YAML or JSON)")
    source.add_argument("--resume", metavar="RUN_DIR", help="fill in what an earlier run is missing")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: %(default)s)")
    p.add_argument("--engines", help="comma-separated subset of the config's engines")
    p.add_argument("--repeats", type=_repeats, help="override the config's repeats (new runs only)")
    p.add_argument("--max-calls", type=int, help="stop after this many calls")
    p.add_argument("--workers", type=int, default=run.DEFAULT_WORKERS)
    p.add_argument("--dry-run", action="store_true",
                   help="answer from bundled fictional fixtures: no API calls, no cost")
    p.add_argument("--no-fetch", action="store_true",
                   help="skip fetching cited pages (faithfulness is then not checked)")
    p.add_argument("--no-raw", action="store_true",
                   help="do not keep each API response as received (q00_r0.raw.json)")
    p.add_argument("--yes", action="store_true", help="do not ask before spending")
    p.set_defaults(handler=cmd_run)

    p = sub.add_parser("report", help="rebuild the report from a run directory")
    p.add_argument("run_dir")
    p.add_argument("--no-fetch", action="store_true")
    p.set_defaults(handler=cmd_report)

    p = sub.add_parser("verify-providers", help="which engine parsers have handled a real response")
    p.set_defaults(handler=cmd_verify_providers)

    p = sub.add_parser("doctor", help="check API keys and, optionally, a config")
    p.add_argument("--config")
    p.set_defaults(handler=cmd_doctor)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.handler(args)
    except (ConfigError, store.PromptSetChanged, FileNotFoundError) as exc:
        _say(f"error: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
