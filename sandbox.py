"""Run model-written code in its own interpreter, with a timeout, without secrets or network.

Candidates are model output, and the CPU correctness check executes every one of them on this
machine, so each run gets:
  * a separate interpreter (`python -I`: no user site, no PYTHON* variables, no script dir on
    sys.path) in a temporary working directory;
  * an environment with API keys, tokens and passwords removed;
  * no network, through a user + network namespace (`unshare -rn`) where the kernel allows it;
  * a CPU-time limit and a wall-clock timeout.
A kernel that hangs, crashes or tries to phone home takes down only its own subprocess.
"""
import os
import re
import resource
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE)
_NETNS: Optional[bool] = None


def scrubbed_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The current environment without anything that looks like a credential."""
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV_RE.search(k)}
    # Triton shells out to its bundled ptxas; keep the running interpreter's bin dir on PATH
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


def network_isolation_available() -> bool:
    """Whether `unshare -rn` works here (unprivileged user namespaces). Checked once."""
    global _NETNS
    if _NETNS is None:
        unshare = shutil.which("unshare")
        try:
            _NETNS = bool(unshare) and subprocess.run([unshare, "-rn", "true"], capture_output=True,
                                                      timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NETNS = False
    return _NETNS


def run_python(script: str, args: List[str], cwd: str, timeout_s: float, extra_env: Optional[Dict[str, str]] = None,
               isolate_network: bool = True) -> subprocess.CompletedProcess:
    """Run `script` with this interpreter under the restrictions above.

    Raises subprocess.TimeoutExpired when the wall-clock limit is hit.
    """
    cmd = [sys.executable, "-I", script, *args]
    if isolate_network and network_isolation_available():
        cmd = [shutil.which("unshare"), "-rn", *cmd]
    cpu_limit = max(10, int(timeout_s * 4))  # interpreter threads add up CPU time faster than wall time

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

    return subprocess.run(cmd, cwd=cwd, env=scrubbed_env(extra_env), capture_output=True, text=True,
                          timeout=timeout_s, preexec_fn=limits)
