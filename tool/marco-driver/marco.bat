@echo off
set SCRIPT_PATH=.
set TEMPORARY_DIR=""

set OUTPUT_FILE=""
set PRINT_HELP="False"
set PRINT_VERSION="False"
set COMPILE_ONLY="False"
set SHOULD_LINK="True"
set OPTIONS=
set INPUT_FILES=
set MODEL=
set END_TIME=
set TEMPORARY_DIR=

call :parse_args %*

if defined MODEL set OPTIONS=%OPTIONS% --model=%MODEL%
if defined END_TIME set OPTIONS=%OPTIONS% --end-time=%END_TIME%

if %PRINT_HELP% == "True" (
    marco-driver --help
    exit /b 0
)

if %PRINT_VERSION% == "True" (
    marco-driver --version
    exit /b 0
)

set TEMPORARY_DIR=%tmp%\test~%random%.tmp
mkdir %TEMPORARY_DIR%

marco-driver %OPTIONS% %INPUT_FILES% -o "%TEMPORARY_DIR%\simulation.obj"

if not %errorlevel% == 0 (
    echo Compilation error
    call :cleanup
    exit /b %errorlevel%
)

if %COMPILE_ONLY% == "True" (
    mv "%TEMPORARY_DIR%%MODEL%/simulation.obj" "%OUTPUT_FILE%.o"
)

if %SHOULD_LINK% == "True" (
    link /nologo /machine:x64 "%TEMPORARY_DIR%\simulation.obj" Kernel32.lib %SCRIPT_PATH%\..\lib\MARCORuntime.lib /out:%OUTPUT_FILE%.exe
)

if not %errorlevel% == 0 (
    echo Link error
    call :cleanup
    exit /b %errorlevel%
)

call :cleanup
exit /b 0

:parse_args
set TMP_ARG=%1
if "%TMP_ARG%" == "" (
    goto :eof
)
:: CASE 1: Compiler option
if not "%TMP_ARG:~0,1%" == "-" goto :end_compiler_option
    :: Output file - extract it into a global variable
    if not "%1" == "-o" goto :end_output_file
        shift
        set OUTPUT_FILE="%1"
        shift
        goto :parse_args
    :end_output_file
    if not "%1" == "--end-time" goto :end_end_time
        shift
        set END_TIME=%1
        shift
        goto :parse_args
    :end_end_time
    if not "%1" == "--model" goto :end_model
        shift
        set MODEL=%1
        shift
        goto :parse_args
    :end_model

    :: This is a regular option - just add it to the list.
    set OPTIONS=%OPTIONS% %1

    if "%1" == "--help" set PRINT_HELP="True"
    if "%1" == "--version" set PRINT_VERSION="True"
    if "%1" == "-c" (
        set COMPILE_ONLY="True"
        set SHOULD_LINK="False"
    )
    if "%1" == "--init-only" set SHOULD_LINK="False"
    if "%1" == "--emit-flattened" set SHOULD_LINK="False"
    if "%1" == "--emit-ast" set SHOULD_LINK="False"
    if "%1" == "--emit-modelica-dialect" set SHOULD_LINK="False"
    if "%1" == "--emit-llvm-dialect" set SHOULD_LINK="False"
    if "%1" == "--emit-llvm-ir" set SHOULD_LINK="False"

shift
goto :parse_args
:end_compiler_option

:: CASE 2: A regular file
if exist "%1" (
    set INPUT_FILES=%INPUT_FILES% %1
    shift
    goto :parse_args
)
:: CASE 3: Unsupported

echo ERROR: unrecognised option format: %1. Perhaps non-existent file?
exit /b

:cleanup
if not %TEMPORARY_DIR% == "" (
    rmdir /s /q %TEMPORARY_DIR%
)