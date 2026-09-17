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

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "python-telegram-bot>=22.8,<23"
])


# =============================================================================
# Configuration
# =============================================================================


HF_TOKEN = CONFIG["HF_TOKEN"]

from huggingface_hub import whoami



user_info = whoami(token=HF_TOKEN)
HF_USER = user_info["name"]
SOURCE_BUCKET = f'{CONFIG["SOURCE_BUCKET"]}'
DEST_BUCKET = f'{HF_USER}/{CONFIG["DEST_BUCKET"]}'

print(f"Source Bucket: {SOURCE_BUCKET}")
print(f"Destination Bucket: {SOURCE_BUCKET}")




#MAX_DIM = 3840 # 4k video

STAGES = [
    [
        "Prepare Environment",
        [
            [
                "APT",
                "cd ~; apt update; apt install -y wget git;"
            ]
        ],
    ],
    [
        "Clone",
        [
            [
                "GIT",
                "git clone https://github.com/hamimmahmud0/Thesis.git"
            ]
        ],
    ],
    [
        "Setup Pipeline",
        [
            [
                "SETUP",
                "/root/Thesis/tools/stabilize-hamim/setup"
            ]
        ],
    ],
    [
        "Run Pipeline",
        [
            [
                "RUN",
                "/root/miniconda3/envs/stabilize/bin/stabilize --help"
            ]
        ],
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
# Telegram Bot
# =============================================================================


import asyncio
import functools
import inspect
import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

from telegram import Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


Handler = Callable[
    ["TelegramBot", Update, ContextTypes.DEFAULT_TYPE],
    Any,
]




import requests


class TelegramBot:
    """
    Telegram bot running completely in its own background thread.

    Handler signature:

        async def handler(bot, update, context):
            ...

    Normal non-async functions are also accepted.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        drop_pending_updates: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self._token = token
        self._drop_pending_updates = drop_pending_updates

        self.logger = logger or logging.getLogger("TelegramBot")

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._application: Optional[Application] = None
        self._async_stop_event: Optional[asyncio.Event] = None

        self._running = threading.Event()
        self._startup_done = threading.Event()
        self._stop_requested = threading.Event()

        self._startup_error: Optional[Exception] = None

        self._lock = threading.RLock()

        # Users discovered from PRIVATE messages.
        self._username_to_chatid: dict[str, int] = {}

        # All chats seen: private chats, groups, etc.
        self._chatids: set[int] = set()

        self._default_chat_id: Optional[int] = None
        self._default_username: Optional[str] = None

        self._command_callbacks: dict[str, Handler] = {}
        self._installed_commands: set[str] = set()

        self._reply_callback: Optional[Handler] = None

    # ================================================================
    # Configuration
    # ================================================================

    def set_token(self, token: str) -> bool:
        """
        Change token while the bot is stopped.
        """
        if self.is_running():
            self.logger.error(
                "Cannot change Telegram token while bot is running."
            )
            return False

        token = token.strip()

        if not token:
            self.logger.error("Telegram token cannot be empty.")
            return False

        self._token = token
        return True

    # ================================================================
    # Users / chat IDs
    # ================================================================

    @staticmethod
    def _normalize_username(username: str) -> str:
        return username.strip().lstrip("@")

    def list_usernames(self) -> list[str]:
        """
        Return private-chat usernames seen since the bot started.
        """
        with self._lock:
            return [
                f"@{username}"
                for username in sorted(self._username_to_chatid)
            ]

    def list_chatids(self) -> list[int]:
        """
        Return every chat ID seen by the bot.
        """
        with self._lock:
            return sorted(self._chatids)

    def get_chat_id(self, username: str) -> Optional[int]:
        username = self._normalize_username(username)

        with self._lock:
            return self._username_to_chatid.get(username)

    def set_default_chat_id(self, chat_id: int) -> None:
        """
        Set recipient used by:

            bot.send_message("hello")
        """
        with self._lock:
            self._default_chat_id = int(chat_id)
            self._default_username = None

    def set_user_to_reply_from_user_name(
        self,
        username: str,
    ) -> bool:
        """
        Set default outgoing recipient using a previously-seen username.

        The user must have messaged the bot privately first.
        """

        username = self._normalize_username(username)

        with self._lock:
            chat_id = self._username_to_chatid.get(username)

            if chat_id is None:
                self.logger.warning(
                    "Unknown Telegram user @%s. "
                    "They must message the bot privately first.",
                    username,
                )
                return False

            self._default_chat_id = chat_id
            self._default_username = username

        return True

    # Shorter alias.
    set_reply_user = set_user_to_reply_from_user_name

    # ================================================================
    # Handlers
    # ================================================================

    def set_handler_to_command(
        self,
        command: str,
        callback: Handler,
    ) -> bool:
        """
        Register/replace a command handler.

        Example:

            bot.set_handler_to_command("status", status_handler)
        """

        command = command.strip().lstrip("/").split("@", 1)[0]

        if not command:
            self.logger.error("Command cannot be empty.")
            return False

        if not callable(callback):
            self.logger.error("Command handler must be callable.")
            return False

        with self._lock:
            self._command_callbacks[command] = callback

            already_installed = (
                command in self._installed_commands
            )

        if already_installed:
            return True

        # Allows new handlers to be added while bot is running.
        if self._loop is not None and self._application is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._install_command_handler,
                    command,
                )
            except Exception:
                self.logger.exception(
                    "Could not install /%s handler.",
                    command,
                )
                return False

        return True

    def set_handler_to_reply(
        self,
        callback: Optional[Handler],
    ) -> bool:
        """
        Handler for ordinary text messages.

        Commands are excluded.

        Pass None to disable.
        """

        if callback is not None and not callable(callback):
            self.logger.error(
                "Reply handler must be callable or None."
            )
            return False

        with self._lock:
            self._reply_callback = callback

        return True

    # ================================================================
    # Start / stop
    # ================================================================

    def start(
        self,
        *,
        wait_until_ready: bool = True,
        timeout: float = 10.0,
    ) -> bool:
        """
        Start Telegram bot in an independent daemon thread.

        Startup/network exceptions are logged instead of being propagated
        into the main application.
        """

        if self._thread and self._thread.is_alive():
            return self.is_running()

        if not self._token:
            self.logger.error(
                "Telegram token has not been configured."
            )
            return False

        self._startup_error = None
        self._startup_done.clear()
        self._running.clear()
        self._stop_requested.clear()

        self._thread = threading.Thread(
            target=self._thread_main,
            name="TelegramBotThread",
            daemon=True,
        )

        self._thread.start()

        if not wait_until_ready:
            return True

        if not self._startup_done.wait(timeout):
            self.logger.error(
                "Telegram bot did not start within %.1f seconds.",
                timeout,
            )
            return False

        if self._startup_error:
            self.logger.error(
                "Telegram startup failed: %s",
                self._startup_error,
            )
            return False

        return self.is_running()

    def stop(self, *, timeout: float = 10.0) -> bool:
        """
        Gracefully stop the Telegram thread.
        """

        self._stop_requested.set()

        loop = self._loop
        stop_event = self._async_stop_event

        if (
            loop is not None
            and stop_event is not None
            and not loop.is_closed()
        ):
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except Exception:
                self.logger.exception(
                    "Failed to signal Telegram shutdown."
                )

        thread = self._thread

        if (
            thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

        if thread and thread.is_alive():
            self.logger.error(
                "Telegram bot thread did not shut down cleanly."
            )
            return False

        return True

    def is_running(self) -> bool:
        return self._running.is_set()

    # ================================================================
    # Sending messages
    # ================================================================

    def send_message(
        self,
        message: str,
        *,
        chat_id: Optional[int] = None,
        username: Optional[str] = None,
        parse_mode: Optional[str] = None,
        wait: bool = False,
        timeout: float = 10.0,
        **kwargs,
    ) -> Optional[Future]:
        """
        Thread-safe send.

        Examples:

            bot.send_message("hello")

            bot.send_message(
                "hello",
                chat_id=12345678,
            )

            bot.send_message(
                "hello",
                username="@john",
            )

        By default sending is NON-BLOCKING.

        Any Telegram/network exception is logged and will not terminate
        your main program.
        """

        target = self._resolve_chat_id(
            chat_id=chat_id,
            username=username,
        )

        if target is None:
            return None

        if (
            not self.is_running()
            or self._loop is None
            or self._application is None
        ):
            self.logger.error(
                "Cannot send message: Telegram bot is not running."
            )
            return None

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._application.bot.send_message(
                    chat_id=target,
                    text=str(message),
                    parse_mode=parse_mode,
                    **kwargs,
                ),
                self._loop,
            )

        except Exception:
            self.logger.exception(
                "Could not schedule Telegram message."
            )
            return None

        if wait:
            try:
                future.result(timeout=timeout)
            except Exception:
                self.logger.exception(
                    "Telegram send failed for chat_id=%s",
                    target,
                )
        else:
            # Consume/log background exceptions so they never become
            # unhandled exceptions.
            future.add_done_callback(
                functools.partial(
                    self._send_future_finished,
                    target,
                )
            )

        return future

    async def reply(
        self,
        update: Update,
        message: str,
        **kwargs,
    ) -> Optional[Message]:
        """
        Convenient reply method for use inside handlers.

        Example:

            await bot.reply(update, "Hello")
        """

        try:
            if update.effective_message is None:
                return None

            return await update.effective_message.reply_text(
                str(message),
                **kwargs,
            )

        except Exception:
            self.logger.exception(
                "Telegram reply failed."
            )
            return None

    # ================================================================
    # Internal thread
    # ================================================================

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()

        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self._async_main()
            )

        except Exception as exc:
            self._startup_error = (
                self._startup_error or exc
            )

            self._startup_done.set()

            self.logger.exception(
                "Telegram bot thread crashed."
            )

        finally:
            self._running.clear()

            # Clean remaining asyncio tasks.
            try:
                pending = asyncio.all_tasks(loop)

                for task in pending:
                    task.cancel()

                if pending:
                    loop.run_until_complete(
                        asyncio.gather(
                            *pending,
                            return_exceptions=True,
                        )
                    )

            except Exception:
                self.logger.exception(
                    "Error cleaning Telegram event loop."
                )

            self._application = None
            self._async_stop_event = None
            self._loop = None

            loop.close()

    async def _async_main(self) -> None:
        app = (
            ApplicationBuilder()
            .token(self._token)
            .build()
        )

        self._application = app
        self._async_stop_event = asyncio.Event()

        with self._lock:
            self._installed_commands.clear()

        # Global PTB error handler.
        app.add_error_handler(
            self._global_error_handler
        )

        # Track users/chat IDs independently of command processing.
        app.add_handler(
            MessageHandler(
                filters.ALL,
                self._track_update,
            ),
            group=-100,
        )

        # Normal text messages.
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._dispatch_reply,
            ),
            group=10,
        )

        with self._lock:
            commands = list(
                self._command_callbacks.keys()
            )

        for command in commands:
            self._install_command_handler(command)

        try:
            await app.initialize()

            if app.updater is None:
                raise RuntimeError(
                    "Telegram polling updater unavailable."
                )

            # Forward polling errors to our global PTB error handler.
            def polling_error(error):
                app.create_task(
                    app.process_error(
                        None,
                        error,
                    )
                )

            await app.updater.start_polling(
                drop_pending_updates=(
                    self._drop_pending_updates
                ),
                error_callback=polling_error,
            )

            await app.start()

            self._running.set()
            self._startup_done.set()

            if self._stop_requested.is_set():
                self._async_stop_event.set()

            await self._async_stop_event.wait()

        except Exception as exc:
            if not self._startup_done.is_set():
                self._startup_error = exc
                self._startup_done.set()

            raise

        finally:
            self._running.clear()

            # Every shutdown operation is independently protected.
            try:
                if (
                    app.updater is not None
                    and app.updater.running
                ):
                    await app.updater.stop()
            except Exception:
                self.logger.exception(
                    "Error stopping Telegram updater."
                )

            try:
                if app.running:
                    await app.stop()
            except Exception:
                self.logger.exception(
                    "Error stopping Telegram application."
                )

            try:
                await app.shutdown()
            except Exception:
                self.logger.exception(
                    "Error shutting down Telegram application."
                )

    # ================================================================
    # Handler internals
    # ================================================================

    def _install_command_handler(
        self,
        command: str,
    ) -> None:
        app = self._application

        if app is None:
            return

        with self._lock:

            if command in self._installed_commands:
                return

            if command not in self._command_callbacks:
                return

            self._installed_commands.add(command)

        app.add_handler(
            CommandHandler(
                command,
                functools.partial(
                    self._dispatch_command,
                    command,
                ),
            ),
            group=0,
        )

    async def _dispatch_command(
        self,
        command: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        with self._lock:
            callback = self._command_callbacks.get(
                command
            )

        if callback is None:
            return

        await self._safe_callback(
            callback,
            update,
            context,
            name=f"/{command}",
        )

    async def _dispatch_reply(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        with self._lock:
            callback = self._reply_callback

        if callback is None:
            return

        await self._safe_callback(
            callback,
            update,
            context,
            name="reply",
        )

    async def _safe_callback(
        self,
        callback: Handler,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        name: str,
    ):
        """
        Critical exception boundary around user handlers.
        """

        try:
            result = callback(
                self,
                update,
                context,
            )

            # Allows both:
            #
            # def handler(...):
            #
            # and
            #
            # async def handler(...):
            #
            if inspect.isawaitable(result):
                await result

        except Exception:
            self.logger.exception(
                "Exception inside Telegram handler %s",
                name,
            )

    async def _track_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        try:
            chat = update.effective_chat
            user = update.effective_user

            if chat is None:
                return

            with self._lock:
                self._chatids.add(chat.id)

                # Only map username -> chat ID for actual private chats.
                if (
                    chat.type == "private"
                    and user is not None
                    and user.username
                ):
                    username = self._normalize_username(
                        user.username
                    )

                    self._username_to_chatid[
                        username
                    ] = chat.id

        except Exception:
            self.logger.exception(
                "Could not track Telegram user/chat."
            )

    async def _global_error_handler(
        self,
        update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """
        Last-resort PTB exception handler.
        """

        error = context.error

        if error is None:
            self.logger.error(
                "Unknown Telegram/PTB error."
            )
            return

        self.logger.error(
            "Unhandled Telegram/PTB exception",
            exc_info=(
                type(error),
                error,
                error.__traceback__,
            ),
        )

    # ================================================================
    # Utility
    # ================================================================

    def _resolve_chat_id(
        self,
        *,
        chat_id: Optional[int],
        username: Optional[str],
    ) -> Optional[int]:

        if chat_id is not None:
            return int(chat_id)

        if username is not None:
            result = self.get_chat_id(username)

            if result is None:
                self.logger.warning(
                    "Unknown private Telegram username %s",
                    username,
                )

            return result

        with self._lock:
            result = self._default_chat_id

        if result is None:
            self.logger.error(
                "No default Telegram recipient. "
                "Call set_default_chat_id() or "
                "set_user_to_reply_from_user_name() first."
            )

        return result

    def _send_future_finished(
        self,
        chat_id: int,
        future: Future,
    ):
        try:
            future.result()

        except Exception:
            self.logger.exception(
                "Background Telegram send failed "
                "for chat_id=%s",
                chat_id,
            )

if "BOT_TOKEN" in CONFIG:
    bot = TelegramBot(CONFIG["BOT_TOKEN"])
    bot.set_default_chat_id(CONFIG["BOT_CHAT_ID"])
    bot.start()



# =============================================================================
# Utility functions
# =============================================================================

def send_bot_message(MESSAGE):
    if "BOT_TOKEN" in CONFIG:
        print("Sending Bot Message")   
        try:
            bot.send_message(MESSAGE)
        except Exception:
            traceback.print_exc()




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

    send_bot_message(f"RUNNING stage: {stage_index}: {stage_name}")


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

        return_code = job["process"].returncode

        if return_code == 0:
            successful.append(job)
        else:
            failed.append(job)

    stage_elapsed = time.time() - stage_start_time

    summary_lines = []

    summary_lines.append("=" * 90)
    summary_lines.append(f"STAGE {stage_index} SUMMARY")
    summary_lines.append("=" * 90)

    for job in jobs:

        return_code = job["process"].returncode

        elapsed = (
            time.time()
            - job["start_time"]
        )

        if return_code == 0:
            status = "SUCCESS"
        else:
            status = f"FAILED ({return_code})"

        line = (
            f"{job['name']:<35}"
            f"{status:<18}"
            f"{format_duration(elapsed):<12}"
            f"{job['log_path']}"
        )

        summary_lines.append(line)

    summary_lines.append("-" * 90)
    summary_lines.append(
        f"Successful : {len(successful)}"
    )
    summary_lines.append(
        f"Failed     : {len(failed)}"
    )
    summary_lines.append(
        f"Total      : {len(jobs)}"
    )
    summary_lines.append(
        f"Stage time : {format_duration(stage_elapsed)}"
    )
    summary_lines.append("=" * 90)

    summary = "\n".join(summary_lines)

    print()
    print(summary)

    send_bot_message(summary)

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

        summary = "SUMMARY" + '\n' + "="*20 + '\n'
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

        if not failed_stages:
            summary += "All stages completed successfully.\n"

        send_bot_message(summary)
        time.sleep(10)
    except Exception:
        print("Error in sending summary via bot")
        send_bot_message("Error in generating summary")
        traceback.print_exc()

    if failed_stages:

        sys.exit(1)

    print(
        "All stages completed successfully."
    )


if __name__ == "__main__":
    main()