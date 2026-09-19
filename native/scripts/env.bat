@echo off
REM Set up the x64 native toolchain and put TensorRT / CUDA DLL directories on PATH.
REM Call it, do not run it:   call native\scripts\env.bat
REM
REM NOTE: this file must stay pure ASCII with CRLF line endings. cmd.exe reads .bat
REM as the OEM code page (GBK here), so non-ASCII bytes get mis-decoded into things
REM like '&', which escapes a REM line and gets executed as a command.
REM
REM TensorRT has no usable Windows wheel on PyPI (pip wheels ship neither the C++
REM headers nor trtexec), so it comes from the official zip:
REM   https://developer.nvidia.com/tensorrt/download/11x
REM   accept the licence -> 11.3.0 -> Windows x64 -> Zip, CUDA 13.x variant
REM   (support matrix footnote f10: built with CUDA 13.4, CUDA 13.x only)
REM Extract it so that native\third_party\TensorRT-<version>\include\NvInfer.h exists.

set "NAILONG_NATIVE=%~dp0.."

REM Keep this out of any parenthesised block: the parens in %ProgramFiles(x86)%
REM break cmd's block parsing.
set "PF86=%ProgramFiles(x86)%"

set "VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if defined VSPATH call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if not defined VSPATH echo [env] MSVC x64 toolchain not found, cl.exe will be unavailable

REM TensorRT ships runtime DLLs in bin\ and import libraries in lib\ - different dirs.
set "NAILONG_TRT_ROOT="
for /d %%d in ("%NAILONG_NATIVE%\third_party\TensorRT-*") do set "NAILONG_TRT_ROOT=%%d"
if defined NAILONG_TRT_ROOT (
    set "PATH=%PATH%;%NAILONG_TRT_ROOT%\bin"
    echo [env] TensorRT bin: %NAILONG_TRT_ROOT%\bin
) else (
    echo [env] no native\third_party\TensorRT-* found - see the note at the top of this file
)

set "NAILONG_CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
if exist "%NAILONG_CUDA_BIN%" (
    set "PATH=%PATH%;%NAILONG_CUDA_BIN%"
    echo [env] CUDA runtime: %NAILONG_CUDA_BIN%
) else (
    echo [env] CUDA bin\x64 not found: %NAILONG_CUDA_BIN%
)
