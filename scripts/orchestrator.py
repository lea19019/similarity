#!/usr/bin/env python3
"""
Simple experiment orchestrator for the circuits_languages replication.

Submits run.sh via sbatch, polls for completion, validates results with
Claude Code headless mode, and retries on failure. State is persisted to
JSON so it can be resumed if interrupted.

Usage:
    python scripts/orchestrator.py [--max-retries N] [--poll-interval SECS] [--dry-run]

Run inside tmux on the login node:
    tmux new -s orchestrator
    cd ~/similarity && source .venv/bin/activate
    python scripts/orchestrator.py 2>&1 | tee logs/orchestrator.log
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
SLURM_SCRIPT = PROJECT_DIR / "run.sh"
STATE_FILE = PROJECT_DIR / "logs" / "orchestrator_state.json"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

# ── Expected outputs (used for quick validation) ────────────────────────────

EXPECTED_FILES = [
    "patching_en.npz",
    "patching_es.npz",
    "dla_en.npz",
    "dla_es.npz",
    "neurons_en.npz",
    "neurons_es.npz",
    "pca_L13H7.npz",
    "steering.npz",
]

EXPECTED_FIGURES = [
    "figures/",  # at least the directory should exist with files
]

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")

# ── State management ────────────────────────────────────────────────────────


def load_state() -> dict:
    """Load orchestrator state from disk, or return a fresh state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        log.info("Resumed state from %s", STATE_FILE)
        return state
    return {
        "status": "pending",  # pending | running | validating | completed | failed
        "job_id": None,
        "attempt": 0,
        "max_retries": 2,
        "history": [],
        "claude_session_id": None,
    }


def save_state(state: dict) -> None:
    """Persist state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── SLURM helpers ───────────────────────────────────────────────────────────


def submit_job(dry_run: bool = False) -> str | None:
    """Submit run.sh via sbatch. Returns the SLURM job ID."""
    if dry_run:
        log.info("[DRY RUN] Would submit %s", SLURM_SCRIPT)
        return "DRY_RUN_12345"

    result = subprocess.run(
        ["sbatch", str(SLURM_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    if result.returncode != 0:
        log.error("sbatch failed: %s", result.stderr.strip())
        return None

    # sbatch output: "Submitted batch job 12345678"
    job_id = result.stdout.strip().split()[-1]
    log.info("Submitted job %s", job_id)
    return job_id


def job_is_running(job_id: str) -> bool:
    """Check if a SLURM job is still in the queue (PENDING or RUNNING)."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if status:
        log.info("Job %s status: %s", job_id, status)
        return True
    return False


def get_job_exit_code(job_id: str) -> int | None:
    """Get the exit code of a completed SLURM job via sacct."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-o", "ExitCode", "-X"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        return None
    # ExitCode format: "0:0" (exit_code:signal)
    try:
        return int(output.split(":")[0])
    except (ValueError, IndexError):
        log.warning("Could not parse exit code from sacct: %r", output)
        return None


def get_job_log_path(job_id: str) -> Path | None:
    """Find the stdout log file for a completed job."""
    # run.sh uses: --output=logs/%x_%j.out
    pattern = LOGS_DIR / f"*_{job_id}.out"
    matches = list(LOGS_DIR.glob(f"*_{job_id}.out"))
    return matches[0] if matches else None


# ── Validation ──────────────────────────────────────────────────────────────


def quick_validate() -> tuple[bool, list[str]]:
    """Check that all expected output files exist and are non-empty."""
    missing = []
    for fname in EXPECTED_FILES:
        path = RESULTS_DIR / fname
        if not path.exists() or path.stat().st_size == 0:
            missing.append(fname)

    # Check that figures directory has at least one file
    fig_dir = RESULTS_DIR / "figures"
    if not fig_dir.exists() or not any(fig_dir.iterdir()):
        missing.append("figures/ (empty or missing)")

    if missing:
        return False, missing
    return True, []


def claude_validate(state: dict) -> tuple[bool, str]:
    """
    Call Claude Code in headless mode to validate results.
    Returns (success, response_text).
    """
    prompt = (
        "The SLURM job for the circuits_languages replication just completed. "
        "Please validate the results:\n"
        "1. Check that all expected .npz files exist in results/ and are non-empty\n"
        "2. Load a few .npz files and sanity-check shapes and value ranges\n"
        "3. Check that figures were generated in results/figures/\n"
        "4. Look at the last 50 lines of the job log for any warnings or errors\n"
        "5. Give a brief summary: PASS if everything looks good, FAIL with details if not"
    )

    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        "20",
        "--allowedTools",
        "Read,Glob,Grep,Bash(python *),Bash(ls *),Bash(tail *),Bash(head *),Bash(cat *),Bash(sacct *)",
    ]

    # Resume session if we have one
    if state.get("claude_session_id"):
        cmd.extend(["--resume", state["claude_session_id"]])

    log.info("Calling Claude for validation...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        log.warning("Claude validation timed out after 600s")
        return False, "Validation timed out"

    if result.returncode != 0:
        log.warning("Claude returned non-zero: %s", result.stderr.strip()[:200])
        return False, f"Claude error: {result.stderr.strip()[:200]}"

    # Parse the last JSON line
    try:
        last_line = result.stdout.strip().splitlines()[-1]
        data = json.loads(last_line)
        response_text = data.get("result", "")
        state["claude_session_id"] = data.get("session_id")
        save_state(state)
    except (json.JSONDecodeError, IndexError):
        log.warning("Could not parse Claude output")
        return False, "Could not parse Claude output"

    passed = "PASS" in response_text.upper() and "FAIL" not in response_text.upper()
    return passed, response_text


def claude_diagnose(state: dict, exit_code: int) -> str:
    """Call Claude to diagnose a job failure."""
    log_path = get_job_log_path(state["job_id"])
    log_hint = f"The job log is at {log_path}" if log_path else "Check logs/ for the job output."

    prompt = (
        f"SLURM job {state['job_id']} FAILED with exit code {exit_code}. "
        f"{log_hint}\n"
        "Please:\n"
        "1. Read the job's stdout and stderr logs\n"
        "2. Identify the root cause of the failure\n"
        "3. If it's a code bug, fix it\n"
        "4. Respond with FIXED if you fixed the issue, or UNFIXABLE if it needs manual intervention"
    )

    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        "30",
        "--allowedTools",
        "Read,Edit,Write,Glob,Grep,"
        "Bash(python *),Bash(ls *),Bash(tail *),Bash(head *),Bash(cat *),Bash(sacct *),Bash(uv *)",
    ]

    if state.get("claude_session_id"):
        cmd.extend(["--resume", state["claude_session_id"]])

    log.info("Calling Claude to diagnose failure...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "UNFIXABLE: Claude diagnosis timed out"

    if result.returncode != 0:
        return f"UNFIXABLE: Claude error: {result.stderr.strip()[:200]}"

    try:
        last_line = result.stdout.strip().splitlines()[-1]
        data = json.loads(last_line)
        response_text = data.get("result", "")
        state["claude_session_id"] = data.get("session_id")
        save_state(state)
        return response_text
    except (json.JSONDecodeError, IndexError):
        return "UNFIXABLE: Could not parse Claude output"


# ── Main loop ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment orchestrator")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=120, help="Seconds between squeue polls")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit jobs")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude validation/diagnosis")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset")

    state = load_state()
    state["max_retries"] = args.max_retries

    log.info("=== Orchestrator started (attempt %d/%d) ===", state["attempt"] + 1, args.max_retries + 1)

    while True:
        # ── Submit ───────────────────────────────────────────────────────
        if state["status"] in ("pending", "retry"):
            state["attempt"] += 1
            log.info("Attempt %d/%d", state["attempt"], args.max_retries + 1)

            job_id = submit_job(dry_run=args.dry_run)
            if job_id is None:
                log.error("Failed to submit job. Exiting.")
                state["status"] = "failed"
                save_state(state)
                sys.exit(1)

            state["job_id"] = job_id
            state["status"] = "running"
            state["history"].append({
                "attempt": state["attempt"],
                "job_id": job_id,
                "submitted_at": datetime.now().isoformat(),
            })
            save_state(state)

        # ── Poll ─────────────────────────────────────────────────────────
        if state["status"] == "running":
            if args.dry_run:
                log.info("[DRY RUN] Simulating job completion")
                state["status"] = "validating"
                save_state(state)
            else:
                while job_is_running(state["job_id"]):
                    log.info("Waiting %ds before next poll...", args.poll_interval)
                    time.sleep(args.poll_interval)

                # Job finished — check exit code
                exit_code = get_job_exit_code(state["job_id"])
                log.info("Job %s finished with exit code: %s", state["job_id"], exit_code)

                if exit_code == 0:
                    state["status"] = "validating"
                else:
                    log.error("Job failed (exit code %s)", exit_code)
                    state["history"][-1]["exit_code"] = exit_code

                    if not args.skip_claude:
                        diagnosis = claude_diagnose(state, exit_code)
                        log.info("Claude diagnosis:\n%s", diagnosis[:500])
                        state["history"][-1]["diagnosis"] = diagnosis[:1000]

                        if "FIXED" in diagnosis.upper() and state["attempt"] <= args.max_retries:
                            log.info("Claude fixed the issue — retrying")
                            state["status"] = "retry"
                            save_state(state)
                            continue

                    if state["attempt"] <= args.max_retries:
                        log.info("Retrying...")
                        state["status"] = "retry"
                        save_state(state)
                        continue
                    else:
                        log.error("Max retries exceeded. Giving up.")
                        state["status"] = "failed"
                        save_state(state)
                        sys.exit(1)

                save_state(state)

        # ── Validate ─────────────────────────────────────────────────────
        if state["status"] == "validating":
            # Quick local check first
            ok, missing = quick_validate()
            if not ok:
                log.warning("Quick validation failed — missing: %s", missing)
                state["history"][-1]["missing_files"] = missing

                if state["attempt"] <= args.max_retries:
                    log.info("Retrying due to missing outputs...")
                    state["status"] = "retry"
                    save_state(state)
                    continue
                else:
                    log.error("Max retries exceeded. Missing files: %s", missing)
                    state["status"] = "failed"
                    save_state(state)
                    sys.exit(1)

            log.info("Quick validation passed — all expected files present")

            # Detailed validation with Claude
            if not args.skip_claude:
                passed, details = claude_validate(state)
                state["history"][-1]["validation"] = details[:1000]
                save_state(state)

                if not passed:
                    log.warning("Claude validation failed:\n%s", details[:500])
                    if state["attempt"] <= args.max_retries:
                        state["status"] = "retry"
                        save_state(state)
                        continue
                    else:
                        log.error("Max retries exceeded after validation failure")
                        state["status"] = "failed"
                        save_state(state)
                        sys.exit(1)

                log.info("Claude validation passed")
            else:
                log.info("Skipping Claude validation (--skip-claude)")

            state["status"] = "completed"
            state["completed_at"] = datetime.now().isoformat()
            save_state(state)

        # ── Done ─────────────────────────────────────────────────────────
        if state["status"] == "completed":
            log.info("=== All experiments completed successfully! ===")
            sys.exit(0)

        if state["status"] == "failed":
            log.error("=== Orchestrator finished with failures ===")
            sys.exit(1)


if __name__ == "__main__":
    main()
