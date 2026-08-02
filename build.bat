@echo off
REM Build script for SdetAI on Windows
REM Requires: Visual Studio 2022 (MSVC) or clang-cl, CMake 3.20+, Git

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo ========================================
echo Building SdetAI - Lightweight Coding Agent
echo ========================================

REM Configure
if not exist build (
    mkdir build
)
cd build

echo [1/3] Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_FLAGS="/MP /bigobj" ^
    %*

if errorlevel 1 (
    echo CMake configure failed!
    exit /b 1
)

echo [2/3] Building...
cmake --build . --config Release --parallel

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo [3/3] Build complete!
echo.
echo Executable: %PROJECT_DIR%build\Release\sdetai_main.exe
echo.

REM Run quick test if executable exists
if exist Release\sdetai_main.exe (
    echo Running quick test...
    Release\sdetai_main.exe --help
)

echo.
echo Done!