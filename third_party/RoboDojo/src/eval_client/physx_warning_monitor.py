"""Capture Omniverse/PhysX log warnings and expose bad env indices.

Carb/Kit's logger writes warnings such as

    [Warning] [omni.physx.plugin] Invalid PhysX transform detected for /World/envs/env_4/...

to the process's stdout (Carb's default console sink writes to fd=1, not
stderr). These warnings are emitted by C++ code, so we cannot intercept
them through Python's logging or `sys.stdout`/`sys.stderr` replacement
alone.

Detection paths (all converge on `_broken_envs`):

  1. fd reader threads: we `dup2(pipe_write, 1/2)` so Carb's writes flow
     through a pipe; reader threads scan each chunk for the PhysX
     warning pattern and update `_broken_envs`. If Kit later steals
     fd=1 (common around `env.close()` / `env.reset()`), the reader
     gets EOF and triggers `_reattach()` to rebuild the pipe.
  2. tail watcher thread: reader threads also mirror every captured
     chunk into a private file under `/dev/shm/`. A separate tail
     thread reads that file and applies the same pattern scan, giving
     an independent backup channel.

Echo/mirror throttling: PhysX can emit the same warning (per-substep
"Invalid PhysX transform" storms, CUDA error-700 floods) thousands of
times per second. Echoing all of it would back up the fd=1 pipe and stall
the Kit process. Reader threads therefore throttle the echo + mirror
copies via a per-reader `_ThrottleState` (keep first occurrence per round,
then at most one per key per `THROTTLE_INTERVAL_S`, with a coalesced
"suppressed N" summary). Detection runs on every line regardless, so
throttling never hides a broken env.

Every eval round must start with `reset()` to clear bookkeeping and
truncate the mirror file (or close+unlink+reopen if truncate fails),
guaranteeing the next round starts from an empty mirror.

`PhysXWarningMonitor.start()` must be called BEFORE `AppLauncher(args_cli)`
so that the Kit process inherits the redirected fds.
"""

import atexit
import fcntl
import os
import re
import sys
import threading
import time


class PhysXBrokenError(Exception):
    """Raised by EvalEnv when one or more envs have triggered the PhysX warning.

    The `broken_envs` attribute holds the env indices that should have their
    current seed/layout abandoned and replaced.
    """

    def __init__(self, broken_envs):
        broken_envs = set(int(i) for i in broken_envs)
        super().__init__(f"PhysX broken envs detected: {sorted(broken_envs)}")
        self.broken_envs = broken_envs


class PhysXFatalError(Exception):
    """Raised when the PhysX simulation has died irrecoverably.

    Triggered by FATAL_PATTERN matches (e.g. ``solveStaticBlock fail to launch
    kernel!!``). At this point per-env recovery is impossible and the only fix
    is a fresh process. ``message`` carries the first matched log line for
    diagnostics.
    """

    def __init__(self, message: str | None = None):
        super().__init__(message or "PhysX fatal kernel failure detected")
        self.message = message


class _ThrottleState:
    """Per-reader, per-round dedup / rate-limit bookkeeping.

    Not thread-safe by design: each reader thread owns one instance, so no
    lock is taken on the hot path. A throttle "round" is bounded by
    PhysXWarningMonitor.reset(); the reader rebuilds a fresh _ThrottleState
    whenever it observes a new generation.
    """

    def __init__(self, interval_s: float, cap: int):
        self._interval = interval_s
        self._cap = cap
        # key -> [last_emit_monotonic, suppressed_count]. A plain dict
        # preserves insertion order and is used as a cheap LRU.
        self._state: dict[bytes, list] = {}

    @staticmethod
    def _summary(count: int, key: bytes) -> bytes:
        """Coalesced 'suppressed N' line for `count` dropped copies of `key`."""
        return b"[PhysXMonitor] suppressed %d repeated msgs for %s\n" % (count, key)

    def decide(self, key: bytes, now: float):
        """Return (emit, summary) for one line.

        emit:    whether this line should be written to echo + mirror.
        summary: an optional coalesced "suppressed N" line (bytes) to emit
                 just before this line, or None.
        """
        rec = self._state.get(key)
        if rec is None:
            # First time this round: always emit and start the window.
            if len(self._state) >= self._cap:
                # Evict the oldest key to bound memory.
                self._state.pop(next(iter(self._state)))
            self._state[key] = [now, 0]
            return True, None

        last_emit, suppressed = rec
        if now - last_emit >= self._interval:
            rec[0] = now
            rec[1] = 0
            # Coalesced summary for everything dropped since last emit.
            return True, (self._summary(suppressed, key) if suppressed > 0 else None)

        # Inside the window: drop from echo/mirror, bump the suppressed count.
        rec[1] = suppressed + 1
        return False, None

    def drain_summaries(self) -> list[bytes]:
        """Flush a coalesced summary for every key still holding a backlog.

        Called by the reader at a round boundary (or EOF) before the table
        is discarded, so the suppressed counts are never silently lost.
        """
        return [self._summary(suppressed, key) for key, (_, suppressed) in self._state.items() if suppressed > 0]


# Matches a single line terminator: "\r\n", "\r", or "\n" (order matters so
# "\r\n" is preferred over a lone "\r").
_LINE_TERMINATORS = re.compile(rb"\r\n|\r|\n")


def _split_log_segments(buf: bytes):
    """Split `buf` into complete (segment, terminator) pairs + remainder.

    Terminator is b"\n", b"\r", or b"\r\n". Carriage-return updates are
    surfaced as their own segments so the reader can pass them through
    immediately: code such as EvalEnv's per-step progress line uses
    ``print(..., end="\\r")`` to update in place, and a newline may never
    arrive. Splitting only on "\n" would trap those bytes in the buffer and
    hide step progress from the terminal.

    Scanning uses a compiled regex (C-level) so the Python loop runs once per
    line, not once per byte. A per-byte Python loop here is far too slow to
    drain the fd=1 pipe during a PhysX warning storm and reintroduces the
    back-pressure stall this module exists to prevent.

    Real Kit/Carb log lines are "\n"-terminated plain text (no embedded
    "\r"), so this never falsely splits a normal log line. The unterminated
    remainder is returned for the next chunk; a lone trailing "\r" is held
    in the remainder in case it is the first half of a "\r\n" straddling the
    chunk boundary.
    """
    segments = []
    n = len(buf)
    last_end = 0
    for m in _LINE_TERMINATORS.finditer(buf):
        term = m.group()
        if term == b"\r" and m.end() == n:
            # Possible first half of a "\r\n" split across chunks: defer it.
            break
        segments.append((buf[last_end : m.start()], term))
        last_end = m.end()
    return segments, buf[last_end:]


class PhysXWarningMonitor:
    # Pattern matched against every captured stdout/stderr line.
    PATTERN = re.compile(rb"Invalid PhysX transform detected for /World/envs/env_(\d+)/")
    # Fatal CUDA / GPU solver kernel failures. Covers the entire family of
    # "GPU <kernel> fail to launch kernel!!" messages emitted right before
    # PhysX gives up and aborts the process. Anchored on the trailing
    # "fail to launch kernel" so it does not match unrelated lines.
    FATAL_PATTERN = re.compile(rb"fail to launch kernel")
    # PhysX UI reports this after the simulation has already been stopped.
    # Reusing the current Kit process is not useful; main.py exits with rc=99
    # so eval_policy.sh can start a fresh process.
    SHELL_RESTART_PATTERN = re.compile(rb"PhysX has reported too many errors, simulation has been stopped\.")
    # A CUDA error 700 kills the PhysX CUDA context outright. Do not wait for
    # the main thread to return to Python: it may be wedged in a C++ call.
    # The watchdog exits rc=99 immediately so eval_policy.sh starts a wholly
    # fresh process instead of main.py doing an os.execv self-restart.
    IMMEDIATE_SHELL_RESTART_PATTERN = re.compile(
        rb"PhysX Internal CUDA error\. Simulation cannot continue! Error code 700!"
    )
    KIT_MS_PATTERN = re.compile(rb"\[([\d,]+)ms\]")

    # When the previous round detected any broken env, reset() sleeps this
    # many seconds before clearing bookkeeping so reader / tail threads
    # have time to drain any in-flight Carb warning bytes still sitting in
    # the OS pipe or the mirror file. Skipped on rounds with no detections.
    DRAIN_DELAY_S: float = 0.3
    # Keep filtering late log bytes for this Kit-time margin after reset().
    # This is intentionally based on Kit's own `[xxxms]` timestamp inside
    # the warning line, not on wall-clock arrival time at the Python reader.
    STALE_KIT_MS_MARGIN: int = 1000

    # Once a fatal kernel failure is detected, main.py is expected to surface
    # PhysXFatalError and os.execv-restart. But a dead CUDA context can wedge
    # the main thread inside a C++ PhysX call (env.close()/reset/step) so it
    # never returns to Python to run that handler, and PhysX may stop the sim
    # without raising SIGABRT -> the process then hangs forever with no
    # restart. The fatal watchdog force-exits with rc=99 (which eval_policy.sh
    # restarts) if the process is still alive this long after the fatal was
    # first observed. Override via ROBODOJO_FATAL_FORCE_EXIT_GRACE_S.
    FATAL_FORCE_EXIT_GRACE_S: float = 45.0

    # Echo/mirror throttling: collapse repeated noisy log lines (e.g. the
    # per-substep "Invalid PhysX transform" storms and the CUDA error-700
    # floods) so the fd=1 pipe never backs up and stalls the Kit process.
    # Detection still runs on every line; only the echo and mirror copies are
    # throttled.
    THROTTLE_INTERVAL_S: float = 1.0  # at most one emit per key per this window
    THROTTLE_KEY_CAP: int = 4096  # LRU bound on the per-reader key table

    # Pre-strip the leading "ISO-8601 [xxx,xxxms] " prefix so lines that differ
    # only by timestamp collapse onto the same throttle key.
    _TS_PREFIX_PATTERN = re.compile(rb"^\S+\s+\[[\d,]+ms\]\s+")

    # Grow each captured pipe from the default 64 KiB so a brief reader stall
    # (e.g. the Python reader waiting on the GIL during Python-heavy scene
    # setup) does not immediately fill the pipe and block the C++ Kit/PhysX
    # threads writing to fd=1. Capped by /proc/sys/fs/pipe-max-size.
    _F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)
    DESIRED_PIPE_BYTES = 8 * 1024 * 1024

    # Carb writes to fd=1 by default; cover fd=2 as well in case future log
    # config routes some channels to stderr.
    _TARGET_FDS = (1, 2)

    # tmpfs path keeps the mirror in RAM (no disk / NFS overhead).
    MIRROR_DIR = "/dev/shm"
    MIRROR_PREFIX = "physx_monitor_"
    # Mirror files older than this are considered orphaned and removed on next start.
    STALE_SECONDS = 24 * 3600

    def __init__(self):
        self._lock = threading.Lock()
        self._broken_envs: set[int] = set()
        self._last_seen_kit_ms: int | None = None
        self._stale_envs: set[int] = set()
        self._stale_kit_ms_le: int | None = None
        self._started = False
        self._shutdown = False

        # Bumped (under _lock) by reset() at every round boundary. Reader
        # threads compare their local copy against this and flush + clear
        # their throttle tables when it changes, so per-round dedup state is
        # reset without reset() reaching into the reader threads.
        self._throttle_generation = 0

        # Fatal state: set once, never cleared by reset() - only a fresh
        # process can clear it (see module docstring).
        self._fatal_event = threading.Event()
        self._fatal_message: str | None = None
        self._fatal_requires_shell_restart = False
        self._fatal_requires_immediate_shell_restart = False

        # fd plumbing
        self._read_fds: list[int] = []
        self._saved_fds: list[int] = []
        self._reader_threads: list[threading.Thread] = []
        # Reverse lookup: target_fd (1 or 2) -> saved (dup'd) fd.
        # Needed by _reattach to keep echoing into the original tty/tee.
        self._saved_fd_by_target: dict[int, int] = {}

        # Mirror file: reader threads append, tail thread reads.
        self._mirror_path: str | None = None
        self._mirror_fp = None  # write handle, opened with buffering=0
        self._tail_fp = None  # read handle, owned by _tail_loop
        self._tail_thread: threading.Thread | None = None
        # Signals the tail thread that the mirror inode was replaced
        # (close+unlink+reopen fallback path in reset()) and the tail
        # must close its old fp and reopen the file by path.
        self._tail_reopen_event = threading.Event()

        # Watchdog that force-restarts the process when a fatal kernel failure
        # is detected but the wedged main thread never acts on it.
        self._fatal_watchdog_thread: threading.Thread | None = None

    def _parse_kit_ms(self, line: bytes) -> int | None:
        ms_match = self.KIT_MS_PATTERN.search(line)
        if ms_match is not None:
            try:
                return int(ms_match.group(1).replace(b",", b""))
            except ValueError:
                return None
        return None

    def _scan_line(self, line: bytes):
        """Run broken-env + fatal detection on one complete log line.

        Cheap substring pre-filters gate the regexes. Returns the transform
        PATTERN match (or None) so the reader can derive its throttle key
        from the same match instead of running a second regex; the tail
        watcher ignores the return value.
        """
        m = None
        if b"Invalid PhysX transform" in line:
            m = self.PATTERN.search(line)
            if m is not None:
                self._record_or_drop_warning(int(m.group(1)), self._parse_kit_ms(line))
        if (
            b"fail to launch kernel" in line
            or b"PhysX has reported too many errors" in line
            or b"PhysX Internal CUDA error" in line
        ):
            self._check_fatal(line)
        return m

    def _emit(self, saved_fd: int, data: bytes) -> None:
        """Write `data` to the saved terminal/tee fd and the mirror file.

        Best-effort on both paths. The mirror write is serialized under
        `_lock` against reset()'s seek+truncate / close+reopen so a
        concurrent round boundary cannot tear the write. Both copies see the
        same throttled byte stream, so the tail watcher's redundant
        detection still observes the first occurrence of each key.
        """
        try:
            os.write(saved_fd, data)
        except OSError:
            pass
        if self._mirror_fp is not None:
            with self._lock:
                if self._mirror_fp is not None:
                    try:
                        self._mirror_fp.write(data)
                    except Exception:
                        pass

    def _enlarge_pipe(self, read_fd: int) -> None:
        """Best-effort: grow the pipe buffer for `read_fd`.

        A larger kernel buffer gives the Python reader slack to catch up
        after a brief GIL stall without back-pressuring the C++ Kit/PhysX
        threads writing to fd=1 (which would otherwise block in pipe_write
        and stall the simulation). Target is DESIRED_PIPE_BYTES, capped by
        /proc/sys/fs/pipe-max-size. Any failure is ignored.
        """
        target = self.DESIRED_PIPE_BYTES
        try:
            with open("/proc/sys/fs/pipe-max-size", "rb") as fp:
                target = min(target, int(fp.read().strip()))
        except Exception:
            pass
        try:
            fcntl.fcntl(read_fd, self._F_SETPIPE_SZ, target)
        except Exception:
            pass

    def _check_fatal(self, line: bytes) -> None:
        """Scan a single log line for the fatal kernel-failure pattern.

        Idempotent: only the first matching line is remembered as the fatal
        message; subsequent matches just keep the event set. Safe to call
        from any reader/tail thread.
        """
        requires_immediate_shell_restart = self.IMMEDIATE_SHELL_RESTART_PATTERN.search(line) is not None
        requires_shell_restart = self.SHELL_RESTART_PATTERN.search(line) is not None or requires_immediate_shell_restart
        if self.FATAL_PATTERN.search(line) is None and not requires_shell_restart:
            return
        try:
            decoded = line.decode("utf-8", errors="replace").strip()
        except Exception:
            decoded = repr(line)
        with self._lock:
            if self._fatal_message is None:
                self._fatal_message = decoded
            if requires_shell_restart:
                self._fatal_requires_shell_restart = True
            if requires_immediate_shell_restart:
                self._fatal_requires_immediate_shell_restart = True
        self._fatal_event.set()

    def _record_or_drop_warning(
        self,
        env_idx: int,
        kit_ms: int | None,
    ) -> None:
        """Record a PhysX warning unless it is stale from the previous round.

        Stale filtering is scoped to env indices that were broken at the
        last reset(). A late-arriving warning with an old Kit timestamp for
        one of those envs is dropped; warnings for other envs, or warnings
        with a later Kit timestamp, still trigger recovery normally.
        """
        with self._lock:
            if kit_ms is not None:
                if self._last_seen_kit_ms is None:
                    self._last_seen_kit_ms = kit_ms
                else:
                    self._last_seen_kit_ms = max(self._last_seen_kit_ms, kit_ms)

            is_stale = (
                env_idx in self._stale_envs
                and kit_ms is not None
                and self._stale_kit_ms_le is not None
                and kit_ms <= self._stale_kit_ms_le
            )
            if not is_stale:
                self._broken_envs.add(env_idx)

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _sweep_stale_mirrors(self) -> None:
        """Remove orphaned mirror files left behind by previous crashed runs.

        Only files matching the mirror naming convention and older than
        STALE_SECONDS are removed. Best-effort: any failure is swallowed.
        """
        now = time.time()
        try:
            entries = os.listdir(self.MIRROR_DIR)
        except OSError:
            return
        removed = 0
        for name in entries:
            if not (name.startswith(self.MIRROR_PREFIX) and name.endswith(".log")):
                continue
            path = os.path.join(self.MIRROR_DIR, name)
            try:
                if now - os.path.getmtime(path) > self.STALE_SECONDS:
                    os.unlink(path)
                    removed += 1
            except FileNotFoundError:
                # Another process raced us; harmless.
                pass
            except OSError:
                # Permission / busy: skip silently, will retry next time.
                pass
        if removed:
            try:
                sys.stderr.write(f"[PhysXMonitor] swept {removed} stale mirror file(s) from {self.MIRROR_DIR}\n")
            except Exception:
                pass

    def start(self, enabled: bool = True) -> None:
        """Tee stdout/stderr through pipes, open mirror file, spin up workers.

        Must be called BEFORE AppLauncher() so the Kit subprocess inherits
        the redirected fds. Idempotent.

        When ``enabled`` is False (e.g. a non-articulation task that does not
        need PhysX log monitoring), this is a hard no-op: no fds are
        redirected and no reader / tail / watchdog threads are spawned, so the
        monitor consumes no resources and stdout/stderr flow straight to the
        original tty/tee. The other public methods stay safe no-ops in this
        state (`_mirror_fp` is None, `_fatal_event` is never set), and
        `add_broken_envs()` keeps working so EvalEnv's NaN pre-check can still
        recover broken envs.
        """
        if self._started:
            return

        if not enabled:
            self._started = True
            return

        # 1) Reap orphaned mirror files from previous crashed runs.
        self._sweep_stale_mirrors()

        # 2) Open this run's own mirror file (RAM-backed, unbuffered writes).
        self._mirror_path = os.path.join(
            self.MIRROR_DIR,
            f"{self.MIRROR_PREFIX}{os.getpid()}_{int(time.time())}.log",
        )
        self._mirror_fp = open(self._mirror_path, "ab", buffering=0)

        # 3) For each target fd, build the pipe and remember the saved fd so we
        #    can both echo back to the original terminal/tee AND reattach later.
        for target_fd in self._TARGET_FDS:
            r, w = os.pipe()
            self._enlarge_pipe(r)
            saved_fd = os.dup(target_fd)
            os.dup2(w, target_fd)
            os.close(w)
            self._saved_fd_by_target[target_fd] = saved_fd
            self._read_fds.append(r)
            self._saved_fds.append(saved_fd)
            self._reader_threads.append(
                threading.Thread(
                    target=self._reader_loop,
                    args=(r, saved_fd, target_fd),
                    name=f"PhysXWarningMonitor-fd{target_fd}",
                    daemon=True,
                )
            )

        # 4) Tail watcher reads the mirror file as a redundant detection path.
        self._tail_thread = threading.Thread(
            target=self._tail_loop,
            args=(self._mirror_path,),
            name="PhysXWarningMonitor-tail",
            daemon=True,
        )

        # 5) Fatal watchdog: force-restarts the process if a detected fatal is
        #    never acted on by a wedged main thread.
        self._fatal_watchdog_thread = threading.Thread(
            target=self._fatal_watchdog_loop,
            name="PhysXWarningMonitor-fatal-watchdog",
            daemon=True,
        )

        self._started = True
        for t in self._reader_threads:
            t.start()
        self._tail_thread.start()
        self._fatal_watchdog_thread.start()

        # Guarantee mirror file is unlinked on interpreter shutdown even if
        # main.py forgets to call shutdown() explicitly.
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # fd reader / self-heal
    # ------------------------------------------------------------------

    def _reader_loop(
        self,
        read_fd: int,
        saved_fd: int,
        target_fd: int = -1,
    ) -> None:
        """Drain one pipe, echo to terminal, mirror to file, scan for PATTERN.

        On EOF the fd has been stolen by Kit/Carb (most common cause). We
        log a warning to stderr and call _reattach() to rebuild the pipe
        and start a fresh reader, so subsequent warnings are still
        captured.

        Detection (`_record_or_drop_warning`) and fatal scanning
        (`_check_fatal`) run on every line unconditionally. The echo and
        mirror copies are throttled via a per-reader `_ThrottleState`:
        repeated noisy lines (per-substep "Invalid PhysX transform" storms,
        CUDA error-700 floods) are coalesced so the fd=1 pipe cannot back up
        and stall the Kit process. Throttle state is per-reader (no shared
        lock on the hot path) and is flushed + rebuilt whenever reset() bumps
        `_throttle_generation`.

        Matches are recorded into `_broken_envs` (a set) unless their Kit
        timestamp falls inside the stale-warning window installed by reset().
        That prevents late-arriving bytes from the previous round from
        leaking into the new round.
        """
        buf = b""
        throttle = _ThrottleState(self.THROTTLE_INTERVAL_S, self.THROTTLE_KEY_CAP)
        local_gen = self._throttle_generation
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                # Pipe was closed under us during shutdown; just exit.
                return

            if not chunk:
                # EOF: write end of the pipe has no more references. Almost
                # always means Kit replaced fd=target_fd with something else.
                # Scan + flush the partial line and any pending suppression
                # summaries so nothing is silently dropped across the reattach.
                if buf:
                    self._process_log_segment(saved_fd, buf, b"", throttle)
                    buf = b""
                for summary in throttle.drain_summaries():
                    self._emit(saved_fd, summary)
                if not self._shutdown:
                    try:
                        sys.stderr.write(f"[PhysXMonitor] reader fd={target_fd} got EOF, reattaching.\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    self._reattach(target_fd)
                return

            # Round boundary: reset() bumped the generation. Flush pending
            # summaries to the terminal/tee, then start a fresh throttle table
            # so the new round re-emits the first occurrence of each key.
            if self._throttle_generation != local_gen:
                for summary in throttle.drain_summaries():
                    self._emit(saved_fd, summary)
                throttle = _ThrottleState(self.THROTTLE_INTERVAL_S, self.THROTTLE_KEY_CAP)
                local_gen = self._throttle_generation

            # Segment-based processing: peel complete lines, keeping the
            # trailing partial in `buf`. Carriage-return progress updates are
            # split out so they reach the terminal immediately.
            buf += chunk
            segments, buf = _split_log_segments(buf)
            for segment, term in segments:
                self._process_log_segment(saved_fd, segment, term, throttle)

    def _reattach(self, target_fd: int) -> None:
        """Rebuild the pipe for `target_fd` after Kit stole it, then spawn a
        fresh reader thread. Safe to call concurrently for different fds.
        """
        if self._shutdown:
            return
        try:
            r, w = os.pipe()
            self._enlarge_pipe(r)
            # Claim target_fd back. This may race with another Kit dup2 -- if
            # so, the next EOF will trigger another reattach.
            os.dup2(w, target_fd)
            os.close(w)
        except OSError as e:
            try:
                sys.stderr.write(f"[PhysXMonitor] reattach pipe failed: {e}\n")
            except Exception:
                pass
            return

        saved_fd = self._saved_fd_by_target.get(target_fd)
        if saved_fd is None:
            # Should not happen; defensive.
            return

        threading.Thread(
            target=self._reader_loop,
            args=(r, saved_fd, target_fd),
            name=f"PhysXWarningMonitor-fd{target_fd}-r",
            daemon=True,
        ).start()

    def _process_log_segment(
        self,
        saved_fd: int,
        segment: bytes,
        term: bytes,
        throttle: _ThrottleState,
    ) -> None:
        """Scan one complete log segment, then emit it through the throttle."""
        if term == b"\r":
            # In-place progress update (e.g. "envN step: i / N\r"). Pass
            # through verbatim; never throttle it and never treat it as a
            # detectable log line, so step progress stays visible.
            self._emit(saved_fd, segment + term)
            return

        # Full log line ("\n" / "\r\n") or final EOF residue (term == b"").
        # Detection runs first and unconditionally so throttling can never hide
        # a broken env or a fatal kernel failure. Reuse its transform match for
        # the throttle key: env-keyed for the "Invalid PhysX transform" storm,
        # else the timestamp-stripped line (which folds the CUDA error-700
        # floods that differ only by timestamp).
        m = self._scan_line(segment)
        if m is not None:
            key = b"invalid_transform:env_" + m.group(1)
        else:
            key = self._TS_PREFIX_PATTERN.sub(b"", segment, count=1)

        emit, summary = throttle.decide(key, time.monotonic())
        if summary:
            self._emit(saved_fd, summary)
        if emit:
            self._emit(saved_fd, segment + term)

    # ------------------------------------------------------------------
    # Tail watcher (redundant detection path via mirror file)
    # ------------------------------------------------------------------

    def _tail_loop(self, path: str) -> None:
        """Continuously tail the mirror file and apply the same PATTERN scan.

        Provides redundancy in case the fd reader thread is temporarily
        blocked (e.g. tee downstream slow) but the mirror write succeeded.
        Also handles inode swaps signaled by `_tail_reopen_event` (set in
        reset()'s close+unlink+reopen fallback path).
        """
        # Wait for the mirror file to materialize (start() opens it before
        # spawning us, so this loop should exit immediately).
        while not os.path.exists(path) and not self._shutdown:
            time.sleep(0.05)
        if self._shutdown:
            return

        fp = open(path, "rb")
        # Start at end-of-file: we only care about new lines from now on.
        fp.seek(0, os.SEEK_END)
        self._tail_fp = fp

        buf = b""
        while not self._shutdown:
            # Handle the close+unlink+reopen fallback from reset(): the
            # on-disk inode at `_mirror_path` was replaced, so our old
            # fp would return EOF forever. Switch to the new inode.
            if self._tail_reopen_event.is_set():
                self._tail_reopen_event.clear()
                try:
                    fp.close()
                except Exception:
                    pass
                try:
                    fp = open(self._mirror_path, "rb")
                    self._tail_fp = fp
                except Exception:
                    # Could not reopen; back off and retry next iteration.
                    self._tail_reopen_event.set()
                    time.sleep(0.05)
                    continue
                buf = b""  # drop any partial line carried over from old fp

            try:
                chunk = fp.read(65536)
            except Exception:
                time.sleep(0.05)
                continue
            if not chunk:
                time.sleep(0.05)
                continue

            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._scan_line(line)

    # ------------------------------------------------------------------
    # Round boundaries
    # ------------------------------------------------------------------

    def _reopen_mirror_locked(self) -> None:
        """Caller MUST hold `self._lock`.

        Close the current mirror_fp, unlink the path, and open a new file
        at the same path. After this returns, `_mirror_fp` either points
        at a fresh empty file or is None (only if the reopen itself fails,
        which is treated as monitor degradation -- the fd reader path still
        works, only the tail backup is lost for this round).
        """
        try:
            if self._mirror_fp is not None:
                self._mirror_fp.close()
        except Exception:
            pass
        self._mirror_fp = None

        try:
            if self._mirror_path and os.path.exists(self._mirror_path):
                os.unlink(self._mirror_path)
        except Exception:
            pass

        try:
            if self._mirror_path:
                self._mirror_fp = open(self._mirror_path, "ab", buffering=0)
        except Exception as e:
            try:
                sys.stderr.write(f"[PhysXMonitor] mirror reopen failed: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass

    def reset(self) -> None:
        """Called at the top of every eval round in main.py.

        Drains obvious stale Invalid-PhysX warnings still sitting in the OS
        pipe buffer or in the mirror file from the previous round, installs
        a Kit-timestamp stale filter for anything that arrives later, then
        clears bookkeeping and resets the mirror file to empty.

        Drain mechanism: PhysX warnings written by Carb at the very end of
        round N may not have reached the reader / tail threads by the time
        reset() is invoked. If we cleared `_broken_envs` immediately, those
        late bytes would land in the freshly-cleared set and trigger a
        false abandon of an unrelated refill seed in round N+1. To prevent
        this, when the previous round saw any broken envs we sleep
        DRAIN_DELAY_S seconds first so reader / tail threads have time to
        consume and parse nearby bytes. Before clearing, we also install a
        stale filter scoped to those broken env indices and bounded by Kit's
        own `[xxxms]` timestamp, so late bytes that arrive after the sleep
        are dropped without suppressing warnings for unrelated envs. Sleep
        is skipped on rounds that produced no detections, so normal
        throughput is unaffected.

        Mirror reset stays atomic w.r.t. reader writes (held under
        `_lock`): fast path is seek+truncate in place; on failure we fall
        back to close+unlink+reopen and signal the tail thread via
        `_tail_reopen_event` to switch to the new inode.
        """
        with self._lock:
            had_broken = bool(self._broken_envs)
        if had_broken:
            time.sleep(self.DRAIN_DELAY_S)

        truncated_in_place = False
        with self._lock:
            previous_broken_envs = set(self._broken_envs)
            if previous_broken_envs and self._last_seen_kit_ms is not None:
                self._stale_envs = previous_broken_envs
                self._stale_kit_ms_le = self._last_seen_kit_ms + self.STALE_KIT_MS_MARGIN
            else:
                self._stale_envs.clear()
                self._stale_kit_ms_le = None
            self._broken_envs.clear()
            # Start a new throttling round; readers will flush their pending
            # suppression summaries and rebuild their throttle tables when
            # they observe this bump.
            self._throttle_generation += 1
            if self._mirror_fp is not None:
                try:
                    self._mirror_fp.seek(0)
                    self._mirror_fp.truncate(0)
                    truncated_in_place = True
                except Exception as e:
                    try:
                        sys.stderr.write(
                            f"[PhysXMonitor] truncate failed ({e}); falling back to close+unlink+reopen.\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
            if not truncated_in_place:
                self._reopen_mirror_locked()
                # Tail's existing fd points at the unlinked-but-still-open
                # old inode (which will get EOF forever). Wake tail up so
                # it closes that fd and reopens by path -> new inode.
                self._tail_reopen_event.set()

        # Fast-path tail rewind: file inode is unchanged, just seek to 0.
        # Outside the lock (only tail thread + reset touch `_tail_fp`;
        # the slow path already handled tail via the event signal).
        if truncated_in_place:
            try:
                if self._tail_fp is not None:
                    self._tail_fp.seek(0)
            except Exception:
                pass

    def get_broken_envs(self) -> set[int]:
        with self._lock:
            return set(self._broken_envs)

    def is_fatal(self) -> bool:
        """Whether a fatal kernel failure has been observed in this process.

        Once true, stays true for the rest of the process lifetime. Only a
        fresh process (e.g. via os.execv from main.py) can clear it.
        """
        return self._fatal_event.is_set()

    def get_fatal_message(self) -> str | None:
        """First matched fatal log line, or None if no fatal has been seen."""
        with self._lock:
            return self._fatal_message

    def requires_shell_restart(self) -> bool:
        """Whether the current fatal condition should bypass os.execv."""
        with self._lock:
            return self._fatal_requires_shell_restart

    def add_broken_envs(self, envs) -> None:
        """Programmatically flag envs as broken.

        Used by EvalEnv's NaN pre-check: when the pre-check detects a
        non-finite endpose it publishes the offending env indices here in
        addition to raising PhysXBrokenError, so main.py's
        except-Exception backstop (which also reads get_broken_envs)
        sees the same picture.
        """
        with self._lock:
            self._broken_envs.update(int(i) for i in envs)

    # ------------------------------------------------------------------
    # Fatal watchdog
    # ------------------------------------------------------------------

    def _fatal_watchdog_loop(self) -> None:
        """Force-restart the process if a detected fatal is never acted on.

        The reader sets ``_fatal_event`` as soon as it sees a "fail to launch
        kernel" line. main.py is expected to then raise PhysXFatalError and
        os.execv-restart. But a dead CUDA context can wedge the main thread
        inside a C++ PhysX call (env.close()/reset/step) so it never returns
        to Python to run that handler, and PhysX may stop the sim without
        raising SIGABRT -> the process hangs forever.

        This watchdog waits for the fatal event, gives the cooperative
        restart path a grace period (os.execv replaces the process image and
        kills this thread, so the happy path never reaches the force-exit),
        and otherwise force-exits with rc=99 -- which eval_policy.sh's retry
        loop restarts into a fresh process. Progress is recovered from the
        resume manifest persisted at the end of each completed batch.
        """
        try:
            grace = float(
                os.environ.get(
                    "ROBODOJO_FATAL_FORCE_EXIT_GRACE_S",
                    self.FATAL_FORCE_EXIT_GRACE_S,
                )
            )
        except (TypeError, ValueError):
            grace = self.FATAL_FORCE_EXIT_GRACE_S

        # Block until a fatal kernel failure is observed (or clean shutdown).
        while not self._shutdown:
            if self._fatal_event.wait(timeout=1.0):
                break
        if self._shutdown:
            return

        # CUDA error 700 explicitly requests an immediate shell-level restart;
        # all other fatal errors retain the cooperative main.py grace period.
        # Re-check the immediate flag while waiting because a regular fatal
        # line may wake the watchdog shortly before the CUDA error-700 line.
        deadline = time.monotonic() + grace
        immediate_shell_restart = False
        while time.monotonic() < deadline:
            if self._shutdown:
                return
            with self._lock:
                immediate_shell_restart = self._fatal_requires_immediate_shell_restart
            if immediate_shell_restart:
                break
            time.sleep(0.5)

        effective_grace = 0.0 if immediate_shell_restart else grace
        try:
            sys.stderr.write(
                f"[PhysXMonitor] fatal kernel failure not handled within "
                f"{effective_grace:.0f}s ({self._fatal_message!r}); force-exiting rc=99 "
                f"so eval_policy.sh restarts a fresh process.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        # Immediate _exit: the main thread is wedged in a C++/CUDA call and
        # cannot be unwound; a normal sys.exit would not interrupt it.
        os._exit(99)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close the mirror file and unlink it from /dev/shm.

        Idempotent; safe to call from both atexit and main.py explicitly.
        """
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self._mirror_fp is not None:
                self._mirror_fp.close()
                self._mirror_fp = None
        except Exception:
            pass
        try:
            if self._mirror_path and os.path.exists(self._mirror_path):
                os.unlink(self._mirror_path)
        except Exception:
            pass


_MONITOR = PhysXWarningMonitor()


def get_monitor() -> PhysXWarningMonitor:
    return _MONITOR
