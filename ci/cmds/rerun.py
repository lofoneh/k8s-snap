#!/usr/bin/env python3
#
# Copyright 2026 Canonical, Ltd.
#
"""
Auto-rerun subcommands for `k8s-ci`.

Automatically re-trigger failed workflow runs whose failures are caused by
known transient infrastructure issues (snap store rate limits, build-container
network flakes, artifact upload timeouts, ...), instead of requiring a human
to click "Re-run failed jobs".

Safety guardrails (all hard stops, evaluated before any rerun):
  * the run must have concluded as 'failure'
  * at most MAX_ATTEMPTS total attempts per run (i.e. 2 automatic reruns)
  * at most --max-daily reruns per UTC day across the whole repository
  * every failed step's log section must contain at least one known transient
    signature AND no hard test-failure indicator -- otherwise no rerun

This module intentionally only implements the rerun *action*. Detection,
classification quality metrics and reporting of flaky failures are handled
separately.
"""

import argparse
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone

LOG = logging.getLogger(__name__)

# Maximum total number of attempts of a workflow run that this automation is
# willing to trigger. Attempt 1 is the original run, so MAX_ATTEMPTS=3 means
# at most 2 automatic reruns.
MAX_ATTEMPTS = 3

# Default cap on the number of automatic reruns per UTC day (repository-wide).
DEFAULT_MAX_DAILY_RERUNS = 20

# File name of the workflow that invokes this command. Its runs are counted to
# enforce the daily rerun budget.
AUTO_RERUN_WORKFLOW_FILE = "auto-rerun-flaky.yaml"

# Known transient infrastructure failure signatures (regular expressions).
#
# A failed run is only rerun automatically if every failed step's log section
# contains at least one of these signatures AND no hard test-failure indicator
# (see HARD_FAILURE_SIGNATURES). Keep this list focused on infrastructure
# issues that are unrelated to the correctness of the code under test. When
# adding a new signature, prefer specific patterns over broad ones.
TRANSIENT_SIGNATURES = [
    # Snap store rate limiting.
    r"HTTP Error 429",
    r"Too Many Requests",
    r"\[429\]",
    # Build container / apt / network flakes.
    r"Failed to fetch .*archive\.ubuntu\.com",
    r"Unable to connect to .*archive\.ubuntu\.com",
    r"Failed to fetch .*security\.ubuntu\.com",
    r"Temporary failure resolving",
    r"Connection timed out",
    r"Connection reset by peer",
    r"Connection refused",
    r"Network is unreachable",
    r"Failed to establish a new connection",
    r"Errno 101",
    r"Errno 111",
    r"Errno 113",
    # HTTP client retry exhaustion around network flakes.
    r"ConnectionError",
    r"MaxRetryError",
    r"NewConnectionError",
    # Go module proxy flakes.
    r"sum\.golang\.org",
    r"proxy\.golang\.org",
    # Artifact upload/download flakes.
    r"Failed to CreateArtifact",
    r"Failed to upload artifact",
    r"Request timeout",
    # Retry-exhaustion wrappers around the above (e.g. tenacity).
    r"tenacity\.RetryError",
    # Launchpad / CLA transient server errors.
    r"Launchpad.*\b5\d\d\b",
    r"\b502 Bad Gateway\b",
    r"\b503 Service Unavailable\b",
    r"\b504 Gateway Time-?out\b",
]

# Signatures of genuine test-logic/assertion failures. If any of these appear
# in a failed step's log section, the failure is attributed to the code under
# test and the run is NOT rerun, even if a transient signature also appears.
HARD_FAILURE_SIGNATURES = [
    r"AssertionError",
    r"^E\s+assert",
    r"\bassert\b.*[=!<>]=.*\b(expected|got|actual)\b",
    r"TestFailure",
    r"^FAIL: ",
    r"expected .* but got",
]

# `gh run view --log-failed` prefixes lines with "<job>\t<step>\t<timestamp> ";
# the per-job logs API prefixes lines with just "<timestamp> ". The first line
# may carry a BOM. Strip either form before matching.
_LINE_PREFIX = re.compile(
    r"^(?:[^\t]*\t[^\t]*\t)?\ufeff?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?"
)

# Matches the timestamp GitHub Actions prepends to every raw log line, e.g.
# "2026-09-19T02:05:44.4655607Z ...". Used to slice a job log by step, since
# steps do not otherwise correspond 1:1 with "##[group]" sections (composite
# actions emit extra groups, and steps after a failure, e.g. artifact
# uploads, emit groups too).
_LINE_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d+Z")


class RerunError(Exception):
    """Fatal error while evaluating or triggering a rerun."""


def _gh(args: list[str]) -> subprocess.CompletedProcess:
    """Run a `gh` CLI command and return the completed process."""
    LOG.debug("running: gh %s", " ".join(args))
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True)


def _gh_api_json(endpoint: str) -> dict:
    """Call the GitHub API via `gh api` and return the parsed JSON body."""
    return json.loads(_gh(["api", endpoint]).stdout)


def _strip_prefix(line: str) -> str:
    return _LINE_PREFIX.sub("", line)


def _failed_step_sections(repo: str, run_id: int, attempt: int) -> list[str] | None:
    """Return the log sections of every failed step of every failed job.

    Returns None if any failed job's logs are unavailable (e.g. expired),
    in which case the caller must not rerun.
    """
    jobs = _gh_api_json(
        f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
    )["jobs"]
    sections: list[str] = []
    for job in jobs:
        if job["conclusion"] != "failure":
            continue
        failed_steps = [s for s in job["steps"] if s["conclusion"] == "failure"]
        if not failed_steps:
            continue
        try:
            log = _gh(["api", f"repos/{repo}/actions/jobs/{job['id']}/logs"]).stdout
        except subprocess.CalledProcessError as e:
            LOG.warning("could not fetch logs for job %s: %s", job["id"], e.stderr)
            return None
        if "BlobNotFound" in log[:200]:
            LOG.warning("logs for job %s have expired", job["id"])
            return None
        lines = log.splitlines()
        for step in failed_steps:
            section = _step_log_section(lines, step)
            if section:
                sections.append(section)
    return sections


def _parse_step_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _parse_line_timestamp(line: str) -> datetime | None:
    m = _LINE_TIMESTAMP.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _step_log_section(lines: list[str], step: dict) -> str:
    """Return the log lines whose timestamp falls within a step's run time.

    Steps don't correspond 1:1 with a job log's "##[group]Run " sections
    (composite actions emit extra groups, and steps that run after a
    failure, e.g. artifact uploads, emit groups of their own), so sections
    are identified via each step's own started_at/completed_at instead.
    The step API only has second precision, so up to a second's worth of
    lines from the immediately adjacent step may bleed in at either
    boundary; harmless in practice since it's a tiny fraction of a step's
    log.
    """
    start = _parse_step_timestamp(step.get("started_at"))
    end = _parse_step_timestamp(step.get("completed_at")) or datetime.max.replace(
        tzinfo=timezone.utc
    )
    if start is None:
        return ""
    matched = [
        line
        for line in lines
        if (ts := _parse_line_timestamp(line)) is not None and start <= ts <= end
    ]
    return "\n".join(matched)


def _section_is_transient(
    section: str,
    transient: list[re.Pattern],
    hard_fail: list[re.Pattern],
) -> bool:
    """Return True if a failed step's log section looks like a transient flake."""
    lines = [_strip_prefix(line) for line in section.splitlines()]
    if not any(p.search(line) for line in lines for p in transient):
        LOG.info("no transient signature found in failed step section")
        return False
    for line in lines:
        if any(p.search(line) for p in hard_fail):
            LOG.info("hard failure signature in failed step: %.200s", line)
            return False
    return True


def _count_reruns_today(repo: str) -> int:
    """Count how many times the auto-rerun workflow has run today (UTC).

    The auto-rerun workflow runs at most once per completed target workflow
    run and triggers at most one rerun per invocation, so the number of its
    own runs today is a conservative upper bound on the number of automatic
    reruns triggered today.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        proc = _gh(
            [
                "run",
                "list",
                "--repo",
                repo,
                "--workflow",
                AUTO_RERUN_WORKFLOW_FILE,
                "--created",
                f">={today}",
                "--json",
                "databaseId",
                "--limit",
                "200",
            ]
        )
    except subprocess.CalledProcessError:
        # The auto-rerun workflow does not exist yet (e.g. first ever run);
        # treat as zero reruns so far today.
        LOG.debug("could not list auto-rerun workflow runs; assuming 0 today")
        return 0
    return len(json.loads(proc.stdout or "[]"))


def cmd_auto_rerun(args: argparse.Namespace) -> int:
    """Evaluate a completed workflow run and rerun it if it is a known flake."""
    repo = args.repo
    run_id = args.run_id
    transient = [re.compile(p) for p in TRANSIENT_SIGNATURES]
    hard_fail = [re.compile(p) for p in HARD_FAILURE_SIGNATURES]

    run = _gh_api_json(f"repos/{repo}/actions/runs/{run_id}")
    conclusion = run.get("conclusion")
    attempt = run.get("run_attempt", 1)
    run_url = run.get("html_url", f"https://github.com/{repo}/actions/runs/{run_id}")

    LOG.info(
        "evaluating run %s (conclusion=%s, attempt=%d)", run_url, conclusion, attempt
    )

    if conclusion != "failure":
        LOG.info("run did not fail (conclusion=%s); nothing to do", conclusion)
        return 0

    if attempt >= MAX_ATTEMPTS:
        LOG.info(
            "run is already at attempt %d (max %d); not rerunning",
            attempt,
            MAX_ATTEMPTS,
        )
        return 0

    reruns_today = _count_reruns_today(repo)
    if reruns_today >= args.max_daily:
        LOG.info(
            "daily auto-rerun budget exhausted (%d >= %d); not rerunning",
            reruns_today,
            args.max_daily,
        )
        return 0

    sections = _failed_step_sections(repo, run_id, attempt)
    if sections is None:
        LOG.info("failed job logs unavailable; cannot verify flake, not rerunning")
        return 0
    if not sections:
        LOG.info("no failed step log sections found; not rerunning")
        return 0
    for section in sections:
        if not _section_is_transient(section, transient, hard_fail):
            LOG.info("not a known transient failure; not rerunning")
            return 0

    if args.dry_run:
        LOG.info("dry-run: would rerun failed jobs of %s", run_url)
        return 0

    LOG.info("all failed steps match known transient signatures; rerunning")
    _gh(["run", "rerun", str(run_id), "--repo", repo, "--failed"])
    LOG.info("rerun triggered for %s", run_url)
    return 0


def _cmd_auto_rerun_checked(args: argparse.Namespace) -> int:
    if not args.repo:
        LOG.error("--repo is required (or set GITHUB_REPOSITORY)")
        return 2
    if not args.run_id:
        LOG.error("--run-id is required (or set WORKFLOW_RUN_ID)")
        return 2
    try:
        return cmd_auto_rerun(args)
    except subprocess.CalledProcessError as e:
        LOG.error("gh command failed: %s\n%s", e.cmd, e.stderr)
        return 1
    except RerunError as e:
        LOG.error("%s", e)
        return 1


def add_rerun_cmds(parser: argparse.ArgumentParser) -> None:
    """
    Register auto-rerun-related subcommands to the given CLI parser.

    Args:
        parser: The parent argparse.ArgumentParser to which subcommands will be added.
    """
    rerun_parser = parser.add_parser(
        "rerun", help="Automatically rerun failed workflows on known transient flakes."
    )
    rerun_sub = rerun_parser.add_subparsers(
        dest="rerun_command", required=True, title="rerun commands"
    )

    p = rerun_sub.add_parser(
        "auto",
        help="Rerun a failed run if all failed steps match known transient signatures.",
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name of the repository (default: GITHUB_REPOSITORY)",
    )
    p.add_argument(
        "--run-id",
        type=int,
        default=int(os.environ.get("WORKFLOW_RUN_ID", "0") or 0),
        help="workflow run ID to evaluate (default: WORKFLOW_RUN_ID)",
    )
    p.add_argument(
        "--max-daily",
        type=int,
        default=int(os.environ.get("MAX_DAILY_AUTO_RERUNS", DEFAULT_MAX_DAILY_RERUNS)),
        help="maximum number of auto-reruns per UTC day "
        f"(default: MAX_DAILY_AUTO_RERUNS or {DEFAULT_MAX_DAILY_RERUNS})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate but do not trigger a rerun",
    )
    p.set_defaults(func=_cmd_auto_rerun_checked)
