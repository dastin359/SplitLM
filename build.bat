@echo off
rem Build trilane against the installed llama.cpp package in ..\llama.cpp\pkg-wos
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 -no_logo || exit /b 1
set PATH=%VS%\VC\Tools\Llvm\ARM64\bin;%PATH%
set ROOT=%~dp0
set PKG=%ROOT%..\llama.cpp\pkg-wos

cmake -S "%ROOT%." -B "%ROOT%build" -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%ROOT%..\llama.cpp\cmake\arm64-windows-llvm.cmake" ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%PKG%" || exit /b 1
cmake --build "%ROOT%build" || exit /b 1
rem run next to the llama/ggml DLLs
copy /y "%ROOT%build\*.exe" "%PKG%\bin\" >nul
copy /y "%ROOT%ui\trilane-ui.html" "%PKG%\bin\" >nul
echo BUILD OK
