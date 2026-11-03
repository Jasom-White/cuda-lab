@echo off
REM standalone CUDA build (no PyTorch). Needs MSVC as nvcc host compiler.
REM usage: build.bat hello.cu -o hello.exe
setlocal EnableExtensions
set "VCVARS="
for %%P in (
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if exist %%P set "VCVARS=%%~P"
if "%VCVARS%"=="" (
  echo [ERROR] vcvars64.bat not found. Install VS Build Tools 2022 with "Desktop development with C++".
  exit /b 1
)
call "%VCVARS%" >nul
set "VSLANG=1033"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
nvcc -arch=sm_120 -O3 %*
