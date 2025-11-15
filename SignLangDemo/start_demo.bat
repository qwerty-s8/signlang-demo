@echo off
setlocal

REM ----- Paths -----
set "DEMO_DIR=%~dp0"
set "VENV_PATH=C:\Users\Lenovo\Downloads\archive (2)\Indian\venv_sign"
set "PYTHON=%VENV_PATH%\Scripts\python.exe"
set "MODEL=%DEMO_DIR%weights\best.onnx"
set "SCRIPT=%DEMO_DIR%webcam_onnx_cls.py"

REM ----- Checks -----
if not exist "%PYTHON%" (
echo Python not found at "%PYTHON%"
echo Fix VENV_PATH in this .bat to your virtualenv path.
pause
exit /b 1
)

if not exist "%SCRIPT%" (
echo Script not found: "%SCRIPT%"
pause
exit /b 1
)

if not exist "%MODEL%" (
echo Model not found: "%MODEL%"
echo Place your ONNX at: %DEMO_DIR%weights\best.onnx
pause
exit /b 1
)

REM ----- Run -----
echo Activating venv and starting demo...
"%PYTHON%" "%SCRIPT%" --model "%MODEL%" --imgsz 224 --smooth 9 --conf_thresh 0.60 --mirror
echo.
echo Exited. Press any key to close.
pause