@echo off
echo Running tests using project virtual environment...
.\venv\Scripts\python -m pytest tests/
if %ERRORLEVEL% NEQ 0 (
    echo Tests failed!
) else (
    echo All tests passed!
)
pause
