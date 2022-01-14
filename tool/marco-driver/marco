#!/usr/bin/env bash

SCRIPT_PATH=$(dirname "${BASH_SOURCE[0]}")
TEMPORARY_DIR=""

INPUT_FILES=()
OPTIONS=()
OUTPUT_FILE=""
PRINT_HELP="False"
PRINT_VERSION="False"
COMPILE_ONLY="False"
SHOULD_LINK="True"

parse_args()
{
  while [ "${1:-}" != "" ]; do
      # CASE 1: Compiler option
      if [[ "${1:0:1}" == "-" ]] ; then
        # Output file - extract it into a global variable
        if [[ "$1" == "-o" ]] ; then
          shift
          OUTPUT_FILE="$1"
          shift
          continue
        fi

        # This is a regular option - just add it to the list.
        OPTIONS+=("$1")

        case $1 in
          --help)
            PRINT_HELP="True"
            ;;
          --version)
            PRINT_VERSION="True"
            ;;
          -c)
            COMPILE_ONLY="True"
            SHOULD_LINK="False"
            ;;
          --init-only | --emit-flattened | --emit-ast | --emit-modelica-dialect | --emit-llvm-dialect | --emit-llvm-ir)
            SHOULD_LINK="False"
            ;;
        esac

        shift
        continue

      # CASE 2: A regular file
      elif [[ -f "$1" ]]; then
        INPUT_FILES+=("$1")
        shift
        continue

      else
        # CASE 3: Unsupported
        echo "ERROR: unrecognised option format: \`$1\`. Perhaps non-existent file?"
        exit 1
      fi
  done
}

cleanup()
{
  if [[ $TEMPORARY_DIR != "" ]]; then
    rm -rf "$TEMPORARY_DIR"
  fi
}

main()
{
  parse_args "$@"

  if [[ $PRINT_HELP == "True" ]]; then
    marco-driver --help
    exit 0
  fi

  if [[ $PRINT_VERSION == "True" ]]; then
    marco-driver --version
    exit 0
  fi

  TEMPORARY_DIR=$(mktemp -d)

  marco-driver "${OPTIONS[@]}" "${INPUT_FILES[@]}" -o "$TEMPORARY_DIR/simulation.o"
  resultCode=$?

  if [ $resultCode -ne 0 ]; then
    echo "Compilation error"
    cleanup
    exit $resultCode
  fi

  if [[ $COMPILE_ONLY == "True" ]]; then
    mv "$TEMPORARY_DIR/simulation.o" "$OUTPUT_FILE.o"
    cleanup
    exit 0
  fi

  if [[ $SHOULD_LINK == "True" ]]; then
    g++ "$TEMPORARY_DIR/simulation.o" $SCRIPT_PATH/../lib/libMARCORuntime.so -o "$OUTPUT_FILE" -Wl,-R$SCRIPT_PATH/../lib
    resultCode=$?

    if [ $resultCode -ne 0 ]; then
      echo "Link error"
      cleanup
      exit $resultCode
    fi
  fi

  cleanup
  exit 0
}

main "${@}"
