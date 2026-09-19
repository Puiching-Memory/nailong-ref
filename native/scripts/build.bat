@echo off
REM Configure and build native/. NMake Makefiles because ninja is not installed here;
REM switch to -G Ninja if you install it.
REM   build.bat            -> Release
REM   build.bat Debug      -> extra arguments are forwarded to cmake
REM
REM This file must stay pure ASCII with CRLF line endings (see native\scripts\env.bat).

call "%~dp0env.bat"

set "NAILONG_BUILD=%~dp0..\build"
cmake -S "%~dp0.." -B "%NAILONG_BUILD%" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release %*
if errorlevel 1 exit /b 1

cmake --build "%NAILONG_BUILD%"
if errorlevel 1 exit /b 1

REM Run the probe immediately: it links TensorRT, so a clean version print proves
REM that headers, import library and runtime DLLs are all actually wired up.
echo.
echo --- trt_probe (toolchain check) ---
"%NAILONG_BUILD%\bin\trt_probe.exe"
if errorlevel 1 (
    echo.
    echo trt_probe exited non-zero - TensorRT DLLs are likely not reachable.
    exit /b 1
)
