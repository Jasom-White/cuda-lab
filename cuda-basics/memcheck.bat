@echo off
REM build + compute-sanitizer memcheck current .cu
setlocal
if "%~1"=="" exit /b 1
call "%~dp0build.bat" "%~1" -o "%~n1.exe"
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe" --tool memcheck "%~dp0%~n1.exe"
