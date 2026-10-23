"""Run Uvicorn with a Windows event loop that supports sockets and subprocesses.

Uvicorn 0.34 selects WindowsSelectorEventLoopPolicy when reload is enabled.
The app's import-time policy change happens after that loop has been created.
Set Proactor before startup and prevent Uvicorn from replacing it; keep all
other Uvicorn CLI options, including reload, unchanged.
"""

import asyncio
import importlib
import logging
import os
import signal
import sys

from uvicorn._subprocess import get_subprocess
from uvicorn.supervisors import ChangeReload


class WindowsReload(ChangeReload):
    """Give each Proactor child a new listening socket after a reload.

    A socket registered with a previous child's IOCP cannot be registered with
    the next child's IOCP. Rebind only after the previous server has exited.
    """

    def restart(self) -> None:
        self.is_restarting = True
        os.kill(self.process.pid, signal.CTRL_C_EVENT)
        # Hidden Windows consoles can ignore CTRL_C_EVENT. A reloader must not
        # retain its listening socket forever while waiting on that process.
        self.process.join(timeout=10)
        if self.process.is_alive():
            import psutil
            logging.getLogger("uvicorn.error").warning(
                "Server did not exit after reload signal; terminating its process tree")
            descendants = psutil.Process(self.process.pid).children(recursive=True)
            for child in descendants:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            self.process.terminate()
            self.process.join(timeout=10)
            psutil.wait_procs(descendants, timeout=10)
        for listener in self.sockets:
            listener.close()
        self.sockets[:] = [self.config.bind_socket()]
        self.process = get_subprocess(config=self.config, target=self.target, sockets=self.sockets)
        self.process.start()


def configure_event_loop() -> bool:
    if sys.platform != "win32":
        return False
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # The installed CLI accepts only auto/asyncio/uvloop. Keep its public
    # "asyncio" value, but disable the policy replacement in this process and
    # its spawned reload children, before Server.run creates a loop.
    from uvicorn.config import LOOP_SETUPS
    LOOP_SETUPS["asyncio"] = None
    return True


def main(argv=None) -> None:
    from uvicorn.main import main as uvicorn_main

    args = list(sys.argv[1:] if argv is None else argv)
    if configure_event_loop():
        importlib.import_module("uvicorn.main").ChangeReload = WindowsReload
        args.extend(["--loop", "asyncio"])
    uvicorn_main(args=args, prog_name="run_backend.py")


if __name__ == "__mp_main__":
    # Windows reload creates a fresh spawned interpreter for each server.
    configure_event_loop()
elif __name__ == "__main__":
    main()
