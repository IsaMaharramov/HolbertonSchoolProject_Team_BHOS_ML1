@echo off
echo ========================================
echo  Seismic First Break Detection Demo
echo  Gradio AI Wrapper
echo ========================================
echo.

cd /d "c:\Users\Admin\Desktop\holb_final\HolbertonSchoolProject_Team_BHOS_ML1"

echo Checking dependencies...
python -c "import gradio, torch, numpy, matplotlib" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] Some dependencies missing!
    echo Installing requirements...
    pip install -r requirements.txt
    echo.
)

echo.
echo Launching Gradio interface...
echo.
echo =========================================
echo  Once started, open your browser to:
echo  http://127.0.0.1:7860
echo =========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python demo_app_2.py

pause
