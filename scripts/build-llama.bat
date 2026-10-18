@echo off
rem Builds the patched llama.cpp (CPU + Adreno OpenCL + Hexagon NPU, signed HTP libs) that SplitLM links against.
rem LLAMA_SRC: llama.cpp checkout with patches\llama.cpp-splitlm.patch applied (default: ..\..\llama.cpp next to this repo).
rem Output: %LLAMA_SRC%\build-wos and %LLAMA_SRC%\pkg-wos. Override the SDK and certificate paths below with env vars.
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 -no_logo || exit /b 1
set PATH=%VS%\VC\Tools\Llvm\ARM64\bin;%PATH%
if not defined LLAMA_SRC set LLAMA_SRC=%~dp0..\..\llama.cpp

if not defined OPENCL_SDK_ROOT set OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2
if not defined HEXAGON_SDK_ROOT set HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0
if not defined HEXAGON_TOOLS_ROOT set HEXAGON_TOOLS_ROOT=%HEXAGON_SDK_ROOT%\tools\HEXAGON_Tools\19.0.07
if not defined HEXAGON_HTP_CERT set HEXAGON_HTP_CERT=%USERPROFILE%\Certs\ggml-htp-v1.pfx
if not defined WINDOWS_SDK_BIN set WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0

cd /d "%LLAMA_SRC%" || exit /b 1
rem these presets add -fno-finite-math-only, which the upstream Windows Snapdragon preset lacks
if not exist CMakeUserPresets.json copy "%~dp0CMakeUserPresets.json" CMakeUserPresets.json >nul
cmake --preset arm64-windows-snapdragon-release -B build-wos || exit /b 1
cmake --build build-wos -j %NUMBER_OF_PROCESSORS% || exit /b 1
cmake --install build-wos --prefix pkg-wos || exit /b 1
echo BUILD OK
