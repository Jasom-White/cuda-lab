@echo off
setlocal EnableExtensions
set "VCVARS="
for %%P in (
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) do if exist %%P set "VCVARS=%%~P"

if "%VCVARS%"=="" (
  echo [ERROR] vcvars64.bat not found.
  echo Install "Build Tools for Visual Studio 2022" with the "Desktop development with C++" workload:
  echo   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  exit /b 1
)

echo [dev] vcvars = "%VCVARS%"
call "%VCVARS%" >nul
set "VSLANG=1033"
set "TORCH_DONT_CHECK_COMPILER_ABI=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%PATH%;%CUDA_HOME%\bin"

if "%~1"=="" (
  echo [dev] env ready. Launching interactive cmd...
  cmd /k
) else (
  python %*
)
