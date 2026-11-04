@echo off
REM Thin wrapper: run.bat <command> [args]   (same commands as python run.py)
where py >nul 2>nul && (py -3 "%~dp0run.py" %*) || (python "%~dp0run.py" %*)
