import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SECRET_NAMES = (
    "GOOGLE_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "INDEX_NAME",
    "LOGIN_PASSWORD",
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_server(tmp_path_factory):
    """Boot tests/smoke_app.py headlessly and yield (url, log_path)."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("streamlit") / "server.log"
    # Never hand real credentials to the no-secret smoke test.
    env = {k: v for k, v in os.environ.items() if k not in SECRET_NAMES}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "tests/smoke_app.py",
                "--server.headless=true",
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        if proc.poll() is not None:
            pytest.fail(f"Streamlit exited early:\n{log_path.read_text()}")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if time.time() > deadline:
            proc.kill()
            pytest.fail(f"Streamlit did not start in 60s:\n{log_path.read_text()}")
        time.sleep(0.5)
    yield url, log_path
    proc.terminate()
    proc.wait(timeout=10)
