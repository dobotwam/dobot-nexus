"""Terminal control for sequential, isolated ATOM D SDK sessions."""

import multiprocessing
import os
import select
import signal
import sys
import termios
import tty
from contextlib import ExitStack

CANCELLED = 3


class OperatorCancelled(Exception):
    pass


def _worker_main(worker, args, operation, stop):
    # Keep terminal signals in the supervisor, including during SDK teardown.
    os.setsid()
    with stop:

        def check_control():
            if stop.poll():
                raise OperatorCancelled("Stopped by operator")

        raise SystemExit(worker(args, operation, check_control))


def _join_worker(child):
    # Usually the pipe cancellation is sufficient. Reap the worker and any
    # policy it started even when a native call blocks during terminal exit.
    child.join(15)
    if child.is_alive():
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                if child.is_alive():
                    os.kill(child.pid, sig)
            child.join(2)
            # A policy could outlive a terminated SDK worker in the same group.
            if sig == signal.SIGTERM:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    if not child.is_alive():
                        break
        child.join()


def operator_session(args, worker):
    fd = sys.stdin.fileno()
    ctx = multiprocessing.get_context("spawn")
    quitting = False

    def request_exit(*_):
        nonlocal quitting
        quitting = True

    def read_keys():
        keys = os.read(fd, 256)
        if not keys or b"\x03" in keys:
            request_exit()
        return keys.lower()

    with ExitStack() as session:
        saved = termios.tcgetattr(fd)
        session.callback(termios.tcsetattr, fd, termios.TCSANOW, saved)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            session.callback(signal.signal, sig, signal.signal(sig, request_exit))
        tty.setcbreak(fd)
        operation = None
        while not quitting:
            if operation is None:
                termios.tcflush(fd, termios.TCIFLUSH)
                print("WAITING  Enter 开始推理 | h 归位 | Ctrl+C 退出", flush=True)
                while not quitting:
                    if not select.select([fd], [], [], 0.1)[0]:
                        continue
                    keys = read_keys()
                    if b"h" in keys:
                        operation = "home"
                        break
                    if b"\r" in keys or b"\n" in keys:
                        operation = "inference"
                        break
            if quitting:
                break

            # Only the parent owns the write end. Its closure also stops a
            # worker when the terminal supervisor unexpectedly disappears.
            recv, send = ctx.Pipe(duplex=False)
            with recv, send:
                child = ctx.Process(target=_worker_main, args=(worker, args, operation, recv))
                child.start()
                recv.close()
                home_requested = False
                try:
                    print("HOMING" if operation == "home" else "INFERRING", flush=True)
                    while not quitting:
                        ready = select.select([fd, child.sentinel], [], [], 0.1)[0]
                        if fd in ready:
                            keys = read_keys()
                            if b"h" in keys and operation == "inference" and not home_requested:
                                send.close()
                                home_requested = True
                                print("STOPPING  推理已收到停止请求，等待控制进程退出", flush=True)
                        if child.sentinel in ready:
                            break
                finally:
                    send.close()
                    _join_worker(child)
                    code = child.exitcode
                    child.close()
            # A failed worker may indicate a hardware fault. Show its result
            # and require a fresh operator command before any further motion.
            operation = "home" if home_requested and code in (0, CANCELLED) else None
        print("EXIT  已退出操作台", flush=True)
    return 130
