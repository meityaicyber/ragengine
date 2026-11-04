#!/usr/bin/env python3
"""One-stop launcher for ragengine.  Standard library only, any OS.

    python run.py setup                 # once: venv, packages, .env, model weights, Ollama model
    python run.py doctor                # what is ready / missing, with the fix for each
    python run.py models                # download the embedding/reranker that .env selects
    python run.py ollama                # pull the chat model that .env selects
    python run.py go   <json-folder>    # ONE COMMAND: installs/downloads anything missing, then
                                        # survey -> validate -> ingest -> crosscheck -> smoke -> serve
    python run.py serve                 # web console at http://localhost:8077

Step by step instead of `go`:
    python run.py survey     <folder>   # field coverage of your JSON, writes nothing
    python run.py validate   <folder>   # what the quality gate would quarantine, writes nothing
    python run.py ingest     <folder>   # build the index (resumable; Ctrl-C is safe)
    python run.py crosscheck <folder>   # stored counts vs raw files
    python run.py smoke                 # a few fixed questions through the whole stack
    python run.py ask "How many high findings are open?" [--trace]
    python run.py chat
    python run.py eval [questions.json]
    python run.py reset                 # wipe the index to start over
    python run.py selftest              # unit tests with fake models (no Ollama / weights needed)

Any other ragaudit CLI command passes straight through, e.g.
    python run.py findings --severity High --status Open
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")

EMBED_REPO = "Qwen/Qwen3-Embedding-0.6B"
RERANK_REPO = "Qwen/Qwen3-Reranker-0.6B"

SMOKE_QUESTIONS = [
    "hello",
    "How many audit reports are indexed?",
    "What types of information are available for each finding?",
    "How many findings are there in total?",
    "Break down the findings by severity",
    "How many high-severity findings are still open?",
    "Which reports failed validation and why?",
    "What access control weaknesses were identified?",
    "Summarise the most severe finding and the recommended fix",
    "Was the HR database encrypted at rest?",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(msg: str) -> None:
    say("\n" + "=" * 72 + f"\n  {msg}\n" + "=" * 72)


def ok(msg: str) -> None:
    say(f"  [ OK ] {msg}")


def bad(msg: str, fix: str = "") -> None:
    say(f"  [FAIL] {msg}")
    if fix:
        say(f"         fix: {fix}")


def warn(msg: str) -> None:
    say(f"  [WARN] {msg}")


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def reexec_in_venv(args: list[str]) -> None:
    """Run this same command with the venv's python, if the venv exists."""
    if in_venv() or not VENV_PY.exists():
        return
    code = subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *args], cwd=ROOT)
    sys.exit(code)


def absolutise(args: list[str]) -> list[str]:
    """Turn arguments that name existing files/folders into absolute paths.

    The launcher always works from the project folder, so a relative path the
    user typed (``./cert_in_json``) must be resolved against where they typed
    it, before changing directory.
    """
    out = []
    for a in args:
        if not a.startswith("-") and len(a) < 1024:
            try:
                p = Path(a).expanduser()
                if p.exists():
                    out.append(str(p.resolve()))
                    continue
            except OSError:
                pass
        out.append(a)
    return out


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith("RAGAUDIT_")})
    return values


def setting(name: str, default: str) -> str:
    return read_env().get(name, default) or default


def cli(*args: str, check: bool = False) -> int:
    """Run `python -m ragaudit.cli ...` from the project root."""
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    code = subprocess.call([py, "-m", "ragaudit.cli", *args], cwd=ROOT)
    if check and code != 0:
        say(f"\n  command failed (exit {code}): ragaudit {' '.join(args)}")
        sys.exit(code)
    return code


def ollama_url() -> str:
    return setting("RAGAUDIT_GENERATION__BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_models() -> list[str] | None:
    """Pulled model names, or None if the server is unreachable."""
    try:
        with urllib.request.urlopen(f"{ollama_url()}/api/tags", timeout=4) as r:
            return [m.get("name", "") for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def model_name() -> str:
    return setting("RAGAUDIT_GENERATION__MODEL_NAME", "qwen3:8b")


def model_pulled(models: list[str], wanted: str) -> bool:
    want = wanted if ":" in wanted else f"{wanted}:latest"
    return any(m == want or m == wanted for m in models)


def ensure_ollama_running() -> bool:
    if ollama_models() is not None:
        return True
    exe = shutil.which("ollama")
    if not exe:
        return False
    say("  starting `ollama serve` in the background ...")
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if IS_WIN:
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([exe, "serve"], **kwargs)
    for _ in range(20):
        time.sleep(1)
        if ollama_models() is not None:
            return True
    return False


def weights_ok(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


def ollama_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return "install from https://ollama.com/download/windows, then reopen the terminal"
    if system == "Darwin":
        return "install from https://ollama.com/download/mac (or `brew install ollama`)"
    return "curl -fsSL https://ollama.com/install.sh | sh"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_setup(argv: list[str]) -> int:
    cpu_torch = "--cpu" in argv
    skip_models = "--no-models" in argv

    head("1/5  Python")
    if sys.version_info < (3, 10):
        bad(f"Python {platform.python_version()} is too old", "install Python 3.10 - 3.12 and re-run")
        return 1
    ok(f"Python {platform.python_version()} ({sys.executable})")

    head("2/5  Virtual environment + packages (first time: several minutes, torch is large)")
    if not VENV_PY.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        ok(f"created {VENV}")
    else:
        ok(f"using existing {VENV}")
    pip = [str(VENV_PY), "-m", "pip"]
    subprocess.check_call([*pip, "install", "--upgrade", "pip"])
    if cpu_torch:
        say("  installing CPU-only torch (smaller download, no CUDA)")
        subprocess.check_call([*pip, "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    subprocess.check_call([*pip, "install", "-r", str(ROOT / "requirements.txt")])
    ok("packages installed")

    head("3/5  Configuration")
    env_file, example = ROOT / ".env", ROOT / ".env.example"
    if not env_file.exists():
        shutil.copy(example, env_file)
        ok("created .env from .env.example (edit it to change model, device, etc.)")
    else:
        ok(".env already exists (left unchanged)")
    device = _detect_device()
    if device != "cpu" and "RAGAUDIT_EMBEDDING__DEVICE=cpu" in env_file.read_text(encoding="utf-8"):
        text = env_file.read_text(encoding="utf-8")
        text = text.replace("RAGAUDIT_EMBEDDING__DEVICE=cpu", f"RAGAUDIT_EMBEDDING__DEVICE={device}")
        text = text.replace("RAGAUDIT_RERANKER__DEVICE=cpu", f"RAGAUDIT_RERANKER__DEVICE={device}")
        env_file.write_text(text, encoding="utf-8")
        ok(f"detected {device}; set embedding/reranker device to {device} in .env")

    head("4/5  Embedding + reranker weights (~2.4 GB total)")
    if skip_models:
        warn("skipped (--no-models). Run `python run.py models` later.")
    else:
        code = cmd_models([])
        if code:
            return code

    head("5/5  Ollama chat model")
    cmd_ollama([])

    head("Setup finished")
    say("  Next:  python run.py doctor")
    say("         python run.py go <path-to-json-folder>")
    return 0


def _detect_device() -> str:
    try:
        out = subprocess.run(
            [str(VENV_PY), "-c",
             "import torch;print('cuda' if torch.cuda.is_available() else "
             "('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu'))"],
            capture_output=True, text=True, timeout=120,
        )
        return (out.stdout.strip() or "cpu").splitlines()[-1]
    except Exception:
        return "cpu"


def active_models() -> dict[str, str]:
    """The models .env selects (or the defaults)."""
    return {
        "chat": model_name(),
        "embedding_repo": setting("RAGAUDIT_EMBEDDING__REPO_ID", EMBED_REPO),
        "embedding_path": setting("RAGAUDIT_EMBEDDING__MODEL_PATH", "models/Qwen3-Embedding-0.6B"),
        "reranker_enabled": setting("RAGAUDIT_RERANKER__ENABLED", "true"),
        "reranker_repo": setting("RAGAUDIT_RERANKER__REPO_ID", RERANK_REPO),
        "reranker_path": setting("RAGAUDIT_RERANKER__MODEL_PATH", "models/Qwen3-Reranker-0.6B"),
    }


def cmd_models(argv: list[str]) -> int:
    """Download the embedding/reranker weights that .env selects."""
    m = active_models()
    targets = [(m["embedding_repo"], ROOT / m["embedding_path"])]
    if m["reranker_enabled"].lower() != "false":
        targets.append((m["reranker_repo"], ROOT / m["reranker_path"]))
    else:
        ok("reranker disabled in .env; not downloading it")
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    for repo, dest in targets:
        if weights_ok(dest):
            ok(f"{repo} already at {dest}")
            continue
        say(f"  downloading {repo} -> {dest}")
        code = subprocess.call([
            py, "-c",
            "import sys\n"
            "try:\n"
            "    from huggingface_hub import snapshot_download\n"
            "    snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])\n"
            "except Exception as exc:\n"
            "    print(f'  download error: {type(exc).__name__}: {str(exc)[:200]}')\n"
            "    sys.exit(1)\n",
            repo, str(dest),
        ])
        if code or not weights_ok(dest):
            bad(f"could not download {repo}",
                f"on a connected machine run:  huggingface-cli download {repo} --local-dir {dest.name}"
                f"  then copy the folder to {dest}")
            return 1
        ok(f"{repo} ready")
    return 0


def cmd_ollama(argv: list[str]) -> int:
    name = model_name()
    if not ensure_ollama_running():
        bad("Ollama is not installed or not running", ollama_install_hint())
        return 1
    ok(f"Ollama reachable at {ollama_url()}")
    models = ollama_models() or []
    if model_pulled(models, name):
        ok(f"model {name} already pulled")
        return 0
    exe = shutil.which("ollama")
    if not exe:
        bad(f"model {name} not pulled and the `ollama` command is not on PATH",
            f"run `ollama pull {name}` in a terminal where ollama works")
        return 1
    say(f"  pulling {name} (several GB on first pull) ...")
    code = subprocess.call([exe, "pull", name])
    if code:
        bad(f"ollama pull {name} failed")
        return 1
    ok(f"model {name} pulled")
    return 0


def cmd_doctor(argv: list[str]) -> int:
    head("Doctor: is everything ready?")
    problems = 0
    m = active_models()
    say("  Active models (edit .env to switch; see the MODELS section at its top):")
    say(f"    chat (Ollama) : {m['chat']}")
    say(f"    embedding     : {m['embedding_repo']}  ->  {m['embedding_path']}")
    say(f"    reranker      : " + (f"{m['reranker_repo']}  ->  {m['reranker_path']}"
                                  if m['reranker_enabled'].lower() != 'false' else "disabled"))
    say("")

    if sys.version_info >= (3, 10):
        ok(f"Python {platform.python_version()}")
    else:
        bad("Python < 3.10", "install Python 3.10+"); problems += 1

    if VENV_PY.exists():
        ok(".venv present")
    else:
        warn(".venv missing; using this Python's packages (python run.py setup creates a venv)")

    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    probe = subprocess.run(
        [py, "-c", "import fastapi, pydantic_settings, numpy, requests, uvicorn;"
                   "import torch, transformers; print(torch.__version__, transformers.__version__,"
                   " 'cuda' if torch.cuda.is_available() else 'cpu')"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        torch_v, tf_v, dev = probe.stdout.split()
        ok(f"packages importable (torch {torch_v}, transformers {tf_v}, {dev} available)")
    else:
        bad("packages missing", "python run.py setup   (or: .venv pip install -r requirements.txt)")
        say("         " + (probe.stderr.strip().splitlines() or ["?"])[-1])
        problems += 1

    if (ROOT / ".env").exists():
        ok(".env present")
    else:
        warn(".env missing, defaults will be used (python run.py setup creates it)")

    for label, key, default in (
        ("embedding weights", "RAGAUDIT_EMBEDDING__MODEL_PATH", "models/Qwen3-Embedding-0.6B"),
        ("reranker weights", "RAGAUDIT_RERANKER__MODEL_PATH", "models/Qwen3-Reranker-0.6B"),
    ):
        path = ROOT / setting(key, default)
        if weights_ok(path):
            ok(f"{label} at {path}")
        elif label.startswith("reranker") and setting("RAGAUDIT_RERANKER__ENABLED", "true").lower() == "false":
            warn(f"{label} missing, but the reranker is disabled in .env")
        else:
            bad(f"{label} missing at {path}", "python run.py models"); problems += 1

    models = ollama_models()
    if models is None:
        bad(f"Ollama not reachable at {ollama_url()}",
            "start the Ollama app, or run `ollama serve` in another terminal"); problems += 1
    else:
        ok(f"Ollama reachable ({len(models)} model(s) pulled)")
        if model_pulled(models, model_name()):
            ok(f"chat model {model_name()} pulled")
        else:
            bad(f"chat model {model_name()} not pulled", f"ollama pull {model_name()}"); problems += 1

    catalog = ROOT / setting("RAGAUDIT_CATALOG_PATH", "var/catalog.sqlite3")
    if catalog.exists():
        import sqlite3
        conn = sqlite3.connect(str(catalog))
        try:
            rows = dict(conn.execute("SELECT ingest_status, COUNT(*) FROM reports GROUP BY 1").fetchall())
            n_find = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            ok(f"catalogue: {rows.get('indexed', 0)} indexed, {rows.get('quarantined', 0)} quarantined "
               f"reports, {n_find} findings")
        except Exception as exc:
            warn(f"catalogue unreadable: {exc}")
        finally:
            conn.close()
    else:
        warn("nothing ingested yet (python run.py ingest <folder>)")

    say("")
    say("  All good." if problems == 0 else f"  {problems} problem(s); apply the fixes above, then re-run doctor.")
    return 0 if problems == 0 else 1


def cmd_smoke(argv: list[str]) -> int:
    """A few fixed questions through the full stack, with the tool trace."""
    head("Smoke test: fixed questions through the full stack")
    say("  Each answer should: give exact numbers for counts, cite sources for narrative")
    say("  questions, and say 'not covered' for the last one instead of guessing.\n")
    failures = 0
    for q in SMOKE_QUESTIONS:
        say("-" * 72)
        say(f"Q: {q}")
        failures += 1 if cli("ask", q, "--trace") else 0
    say("-" * 72)
    return 1 if failures else 0


_PROBE = "import fastapi, pydantic_settings, numpy, requests, uvicorn, torch, transformers, huggingface_hub"


def packages_ok(python: str) -> bool:
    return subprocess.run([python, "-c", _PROBE], capture_output=True).returncode == 0


def auto_prepare() -> bool:
    """Make everything ready, fixing what can be fixed. Returns True when ready.

    Installs missing packages, creates .env, downloads missing model weights
    and pulls the chat model. The two things it cannot do for you are
    installing Python itself and installing the Ollama application.
    """
    head("Preparing: installing / downloading anything that is missing")
    if not packages_ok(sys.executable):
        say("  packages missing; installing requirements.txt (first time: several minutes) ...")
        code = subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        if code or not packages_ok(sys.executable):
            bad("could not install the Python packages", "check your internet connection, then re-run")
            return False
    ok("Python packages")

    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copy(ROOT / ".env.example", env_file)
        ok("created .env from .env.example")
    else:
        ok(".env")

    m = active_models()
    needed = [ROOT / m["embedding_path"]]
    if m["reranker_enabled"].lower() != "false":
        needed.append(ROOT / m["reranker_path"])
    if not all(weights_ok(p) for p in needed):
        say("  model weights missing; downloading (about 2.4 GB for the default models) ...")
        if cmd_models([]) != 0:
            return False
    else:
        ok("embedding / reranker weights")

    if not ensure_ollama_running():
        bad("Ollama is not installed or not running (this one step cannot be automated)",
            ollama_install_hint() + "; then re-run this command")
        return False
    if not model_pulled(ollama_models() or [], model_name()):
        if cmd_ollama([]) != 0:
            return False
    else:
        ok(f"chat model {model_name()}")
    return cmd_doctor([]) == 0


def cmd_go(argv: list[str]) -> int:
    """One command: prepare everything, then survey/validate/ingest/crosscheck/smoke/serve.

    With no folder, it prepares and starts the console on what is already indexed.
    """
    folder = argv[0] if argv and not argv[0].startswith("-") else None
    serve = "--no-serve" not in argv
    force = ["--force"] if "--force" in argv else []
    if folder and not Path(folder).exists():
        say(f"folder not found: {folder}")
        return 2

    if not auto_prepare():
        say("\n  Not ready: apply the fix shown above, then run the same command again.")
        return 1

    if folder is None:
        if serve:
            head("Web console (no folder given: serving what is already indexed)")
            return cli("serve")
        return 0

    head(f"Survey: what does {folder} look like?")
    cli("survey", folder)

    head("Validate: what would be quarantined? (summary)")
    cli("validate", folder, "--quiet", "--only-failing")

    head("Ingest (resumable: if this stops, run the same command again)")
    code = cli("ingest", folder, *force)
    if code != 0:
        say("\n  Ingest reported failures (see 'failed' above). Continuing with what was indexed.")

    head("Crosscheck: stored counts vs raw JSON")
    cli("crosscheck", folder)

    cmd_smoke([])

    if serve:
        head("Web console")
        return cli("serve")
    return 0


def cmd_eval(argv: list[str]) -> int:
    questions = argv[0] if argv else str(ROOT / "eval" / "sample_questions.json")
    return cli("evaluate", "--questions", questions, *argv[1:])


def cmd_reset(argv: list[str]) -> int:
    return cli("reset", *(argv or []))


def cmd_selftest(argv: list[str]) -> int:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    return subprocess.call([py, "-m", "pytest", "-q", str(ROOT / "tests"), *argv], cwd=ROOT)


def cmd_demo(argv: list[str]) -> int:
    """Ingest the bundled fictional sample reports and run the scripted walkthrough."""
    cli("ingest", str(ROOT / "sample_data"))
    return cli("demo", "run")


COMMANDS = {
    "setup": cmd_setup,
    "models": cmd_models,
    "ollama": cmd_ollama,
    "doctor": cmd_doctor,
    "go": cmd_go,
    "smoke": cmd_smoke,
    "eval": cmd_eval,
    "reset": cmd_reset,
    "selftest": cmd_selftest,
    "demo": cmd_demo,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        say(__doc__)
        return 0
    command, argv = sys.argv[1], absolutise(sys.argv[2:])
    os.chdir(ROOT)
    # One-command start: with no venv and no packages, "go" first runs the full
    # setup (creates .venv, installs, downloads weights, pulls the chat model),
    # then continues inside the new venv.
    if command == "go" and not VENV_PY.exists() and not packages_ok(sys.executable):
        say("First run: setting everything up (this happens once).")
        if cmd_setup([]) != 0:
            say("\n  Setup did not finish; fix the problem shown above and run the same command again.")
            return 1
    # Setup must run with the system python (it creates the venv).
    if command not in ("setup",):
        reexec_in_venv([command, *argv])
    if command in COMMANDS:
        return COMMANDS[command](argv)
    # Everything else is a ragaudit CLI command (survey, validate, ingest, ask, serve, ...).
    return cli(command, *argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\ninterrupted")
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        say(f"\n  A step failed: {' '.join(str(a) for a in exc.cmd)} (exit {exc.returncode}).")
        say("  Check the output above (usually the internet connection), then run the same command again.")
        sys.exit(1)
