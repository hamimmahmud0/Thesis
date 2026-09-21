#!/usr/bin/env python3

import os
import signal
import subprocess
import threading
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests
import traceback

os.chdir(os.path.expanduser("~"))




# =============================================================================
# Configuration
# =============================================================================


HF_TOKEN = CONFIG["HF_TOKEN"]

from huggingface_hub import whoami



user_info = whoami(token=HF_TOKEN)
HF_USER = user_info["name"]
DEST_BUCKET = f'{HF_USER}/{CONFIG["DEST_BUCKET"]}'

print(f"Destination Bucket: {DEST_BUCKET}")

VIDEO_FILE_NAME = CONFIG["VIDEO"].split("/")[-1]
RUN_NAME = VIDEO_FILE_NAME.split('.')[0]

#MAX_DIM = 3840 # 4k video

STAGES = [
    [
        "Prepare Environment",
        [
            [
                "Install",
                "cd ~; apt update; apt install -y btop nvtop wget git ffmpeg"
            ]
        ],
    ],

    [
        "Install lgstab",
        [
            [
                "git",
                "git clone https://github.com/hamimmahmud0/costab.git;"
                "cd costab && pip install -e ."
            ]
        ],
    ],

    [
        "Download",
        [
            [
                "wget",
                f"wget {CONFIG["VIDEO"]} -O {VIDEO_FILE_NAME}"
            ]
        ],
    ],
    [
        "run lgstab",
        [
            [
                "lgstab",
                f'lgstab -i {VIDEO_FILE_NAME} -r {RUN_NAME} --devices cuda:0 cuda:1 --crf {CONFIG["CRF"]}'
            ]
        ]
    ],
    [
        "Save to bucket",
        [
            [
                "hf",
                f'export HF_TOKEN={HF_TOKEN} && hf buckets create {DEST_BUCKET} --exist-ok && hf buckets sync runs/. hf://buckets/{DEST_BUCKET}'
            ]
        ]
    ]
]


# Directory where logs will be written
LOG_DIR = Path("logs")

# If True:
#   if any command in a stage fails, later stages WILL NOT run.
#
# If False:
#   later stages will still run even if something failed.
STOP_ON_STAGE_FAILURE = True

# How long to wait after SIGTERM before force-killing jobs
TERMINATION_TIMEOUT = 5


# =============================================================================
# Utility functions
# =============================================================================

def send_bot_message(MESSAGE):
    if "BOT_TOKEN" in CONFIG:
        url = f"https://api.telegram.org/bot{CONFIG["BOT_TOKEN"]}/sendMessage"

        data = {
            "chat_id": CONFIG["CHAT_ID"],
            "text": MESSAGE
        }

        response = requests.post(url, data=data)

        if response.status_code == 200:
            print("Notification sent successfully!")
        else:
            print("Failed:", response.text)


def sanitize_filename(name: str) -> str:
    """
    Convert a name into a safe filename.
    """

    result = "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in name
    )
    return result.strip("_") or "job"


def format_duration(seconds: float) -> str:

    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def terminate_process(process):
    """
    Send SIGTERM to the entire process group.
    """

    if process.poll() is not None:
        return

    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGTERM,
        )
    except ProcessLookupError:
        pass


def kill_process(process):
    """
    Force-kill the entire process group.
    """

    if process.poll() is not None:
        return

    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGKILL,
        )
    except ProcessLookupError:
        pass


def stop_all_jobs(jobs):
    """
    Gracefully terminate all currently running jobs,
    then force kill remaining processes.
    """

    running_jobs = [
        job
        for job in jobs
        if job["process"].poll() is None
    ]

    if not running_jobs:
        return

    print()
    print("Stopping running jobs...")

    # First try SIGTERM
    for job in running_jobs:
        terminate_process(job["process"])

    deadline = time.time() + TERMINATION_TIMEOUT

    while time.time() < deadline:

        if all(
            job["process"].poll() is not None
            for job in running_jobs
        ):
            break

        time.sleep(0.2)

    # SIGKILL anything still alive
    for job in running_jobs:

        if job["process"].poll() is None:
            kill_process(job["process"])


# Lock console writes so output from parallel jobs does not corrupt each line.
PRINT_LOCK = threading.Lock()


def stream_process_output(process, job_name, log_file):
    """
    Stream a process output to both the terminal and its log file in real time.
    Each terminal line is prefixed with the command/job name.
    """
    try:
        for line in iter(process.stdout.readline, ""):
            # Save raw command output to the log file.
            log_file.write(line)
            log_file.flush()

            # Show the same output live in the terminal.
            with PRINT_LOCK:
                print(f"[{job_name}] {line}", end="", flush=True)
    finally:
        if process.stdout is not None:
            process.stdout.close()


# =============================================================================
# Stage execution
# =============================================================================

def run_stage(stage_index, stage_name, commands):
    """
    Run every command in a stage concurrently.

    Returns:
        True  -> every command succeeded
        False -> one or more commands failed
    """

    print()
    print("=" * 90)
    print(
        f"STAGE {stage_index}: {stage_name}"
    )
    print("=" * 90)
    print(
        f"Launching {len(commands)} command(s) in parallel..."
    )
    print()

    try:
        send_bot_message(f"RUNNING stage: {stage_index}: {stage_name}")
    except Exception:
        pass

    stage_start_time = time.time()

    # -------------------------------------------------------------------------
    # Stage log directory
    # -------------------------------------------------------------------------

    safe_stage_name = sanitize_filename(stage_name)

    stage_log_dir = (
        LOG_DIR
        / f"stage_{stage_index:02d}_{safe_stage_name}"
    )

    stage_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jobs = []

    # -------------------------------------------------------------------------
    # Start every command
    # -------------------------------------------------------------------------

    for command_index, item in enumerate(
        commands,
        start=1,
    ):

        if len(item) != 2:
            raise ValueError(
                f"Invalid command in stage "
                f"{stage_index}, command {command_index}.\n"
                f"Expected:\n"
                f"    ['command name', 'command']\n"
                f"Got:\n"
                f"    {item}"
            )

        name, command = item

        safe_name = sanitize_filename(name)

        log_path = (
            stage_log_dir
            / f"{command_index:02d}_{safe_name}.log"
        )

        log_file = open(
            log_path,
            "w",
            buffering=1,
            encoding="utf-8",
        )

        print(
            f"[START] {name}"
        )
        print(
            f"        PID:     pending"
        )
        print(
            f"        Command: {command}"
        )
        print(
            f"        Log:     {log_path}"
        )
        print()

        # Important:
        #
        # Bash executes the entire command string.
        #
        # Therefore this works:
        #
        #   cd project;
        #   source ~/.bashrc;
        #   conda activate myenv;
        #   python train.py
        #
        # Pipes, &&, ||, redirects, variables, etc.
        # also work.
        #
        # start_new_session=True creates a new process
        # group so Ctrl+C can terminate all children.
        process = subprocess.Popen(
            [
                "bash",
                "-lc",
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        # Tee the process output to both terminal and log file in real time.
        output_thread = threading.Thread(
            target=stream_process_output,
            args=(process, name, log_file),
            daemon=True,
        )
        output_thread.start()

        print(
            f"        Started PID: {process.pid}"
        )
        print()

        jobs.append(
            {
                "name": name,
                "command": command,
                "process": process,
                "log_file": log_file,
                "log_path": log_path,
                "start_time": time.time(),
                "reported": False,
                "output_thread": output_thread,
            }
        )

    print("-" * 90)
    print(
        f"Stage {stage_index}: all commands launched."
    )
    print(
        "Waiting for all commands in this stage..."
    )
    print("-" * 90)

    # -------------------------------------------------------------------------
    # Wait for all jobs
    # -------------------------------------------------------------------------

    try:

        while True:

            all_finished = True

            for job in jobs:

                process = job["process"]

                return_code = process.poll()

                if return_code is None:

                    all_finished = False
                    continue

                if job["reported"]:
                    continue

                elapsed = (
                    time.time()
                    - job["start_time"]
                )

                if return_code == 0:

                    status = "SUCCESS"

                else:

                    status = (
                        f"FAILED "
                        f"(exit={return_code})"
                    )

                print(
                    f"[{status}] "
                    f"{job['name']} "
                    f"[{format_duration(elapsed)}]"
                )

                job["reported"] = True

            if all_finished:
                break

            time.sleep(0.5)

    except KeyboardInterrupt:

        print()
        print(
            "Ctrl+C received."
        )

        stop_all_jobs(jobs)

        raise

    finally:

        for job in jobs:

            # Ensure all remaining buffered output has been printed/logged
            # before closing the log file.
            try:
                job["output_thread"].join(timeout=2)
            except Exception:
                pass

            try:
                job["log_file"].close()

            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Stage summary
    # -------------------------------------------------------------------------

    successful = []
    failed = []

    for job in jobs:

        return_code = (
            job["process"].returncode
        )

        if return_code == 0:
            successful.append(job)

        else:
            failed.append(job)

    stage_elapsed = (
        time.time()
        - stage_start_time
    )

    print()
    print("=" * 90)
    print(
        f"STAGE {stage_index} SUMMARY"
    )
    print("=" * 90)

    for job in jobs:

        return_code = (
            job["process"].returncode
        )

        elapsed = (
            time.time()
            - job["start_time"]
        )

        if return_code == 0:

            status = "SUCCESS"

        else:

            status = (
                f"FAILED ({return_code})"
            )

        print(
            f"{job['name']:<35}"
            f"{status:<18}"
            f"{format_duration(elapsed):<12}"
            f"{job['log_path']}"
        )

    print("-" * 90)

    print(
        f"Successful : {len(successful)}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    print(
        f"Total      : {len(jobs)}"
    )

    print(
        f"Stage time : "
        f"{format_duration(stage_elapsed)}"
    )

    print("=" * 90)

    return len(failed) == 0


# =============================================================================
# Main
# =============================================================================

def main():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not STAGES:

        print(
            "No stages configured."
        )

        return

    overall_start_time = time.time()

    completed_stages = 0
    failed_stages = []

    print()
    print("=" * 90)
    print("PARALLEL STAGE COMMAND RUNNER")
    print("=" * 90)

    print(
        f"Stages: {len(STAGES)}"
    )

    print(
        f"Logs:   {LOG_DIR.resolve()}"
    )

    print(
        f"Stop on stage failure: "
        f"{STOP_ON_STAGE_FAILURE}"
    )

    print("=" * 90)

    # -------------------------------------------------------------------------
    # Stages run sequentially
    # -------------------------------------------------------------------------

    try:

        for stage_index, stage in enumerate(
            STAGES,
            start=1,
        ):

            if len(stage) != 2:

                raise ValueError(
                    f"Invalid stage #{stage_index}.\n"
                    f"Expected:\n"
                    f"[\n"
                    f"    'Stage Name',\n"
                    f"    [commands...]\n"
                    f"]"
                )

            stage_name, commands = stage

            if not commands:

                print(
                    f"\n[SKIP] Stage "
                    f"{stage_index}: "
                    f"{stage_name} "
                    f"contains no commands."
                )

                completed_stages += 1
                continue

            success = run_stage(
                stage_index,
                stage_name,
                commands,
            )

            completed_stages += 1

            if not success:

                failed_stages.append(
                    (
                        stage_index,
                        stage_name,
                    )
                )

                if STOP_ON_STAGE_FAILURE:

                    print()
                    print(
                        "A command failed in "
                        f"Stage {stage_index}."
                    )

                    print(
                        "STOP_ON_STAGE_FAILURE=True"
                    )

                    print(
                        "Later stages will not run."
                    )

                    break

            # Only reaches here once every command
            # in this stage has finished.
            if (
                stage_index
                < len(STAGES)
            ):

                print()
                print(
                    f"Stage {stage_index} finished."
                )

                print(
                    "Starting next stage..."
                )

    except KeyboardInterrupt:

        print()
        print("=" * 90)
        print(
            "Execution interrupted by user."
        )
        print("=" * 90)

        sys.exit(130)

    # -------------------------------------------------------------------------
    # Overall summary
    # -------------------------------------------------------------------------

    overall_elapsed = (
        time.time()
        - overall_start_time
    )

    print()
    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(
        f"Configured stages : "
        f"{len(STAGES)}"
    )

    print(
        f"Executed stages   : "
        f"{completed_stages}"
    )

    print(
        f"Failed stages     : "
        f"{len(failed_stages)}"
    )

    print(
        f"Total runtime     : "
        f"{format_duration(overall_elapsed)}"
    )

    if failed_stages:

        print()
        print(
            "Stages containing failures:"
        )

        for (
            stage_index,
            stage_name,
        ) in failed_stages:

            print(
                f"  - Stage "
                f"{stage_index}: "
                f"{stage_name}"
            )

    print()
    print(
        f"Logs are available at:"
    )

    print(
        f"  {LOG_DIR.resolve()}"
    )

    print("=" * 90)

    try:


        if failed_stages:
            summary += "\nStages containing failures:\n"

            for (
                stage_index,
                stage_name,
            ) in failed_stages:
                summary += (
                    f"  - Stage "
                    f"{stage_index}: "
                    f"{stage_name}\n"
                )

        summary += (
            "\n"
            "Logs are available at:\n"
            f"  {LOG_DIR.resolve()}\n"
            + "=" * 90
            + "\n"
        )

        if not failed_stages:
            summary += "All stages completed successfully.\n"

        send_bot_message(summary)
    except Exception:
        print("Error in sending summary via bot")
        traceback.print_exc()

    if failed_stages:

        sys.exit(1)

    print(
        "All stages completed successfully."
    )


if __name__ == "__main__":
    main()