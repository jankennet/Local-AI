@echo off
if "%LLM_API_KEY%"=="" (
    echo LLM_API_KEY is not set. Set it first, e.g.:
    echo   set LLM_API_KEY=your-generated-key-here
    exit /b 1
)

call venv\Scripts\activate
python run.py