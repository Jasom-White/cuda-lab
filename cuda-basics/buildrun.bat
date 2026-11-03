@echo off
REM build + run current .cu without PowerShell "&&" (PS 5.1 does not support it)
setlocal
if "%~1"=="" (
  echo usage: buildrun.bat file.cu
  exit /b 1
)
call "%~dp0build.bat" "%~1" -o "%~n1.exe"
if errorlevel 1 (
  echo [build failed]
  exit /b 1
)
"%~dp0%~n1.exe"
