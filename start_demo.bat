@echo off
setlocal
cd /d "%~dp0"
echo Working dir: %CD%

call "%~dp0.venv\Scripts\activate.bat" || (echo VENV not found & pause & exit /b 1)
where python || (echo Python not found in PATH & pause & exit /b 1)

if not exist ".\webcam_onnx_cls.py" echo webcam_onnx_cls.py not found & pause & exit /b 1
if not exist ".\weights\best.onnx" echo weights\best.onnx not found & pause & exit /b 1

echo Starting demo...
python ".\webcam_onnx_cls.py" ^
  --model ".\weights\best.onnx" ^
  --imgsz 224 ^
  --camera 0 ^
  --labels "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,1,2,3,4,5,6,7,8,9" ^
  --smooth 7 ^
  --conf_thresh 0.50 ^
  --mirror

echo Exit code: %ERRORLEVEL%
pause
endlocal
