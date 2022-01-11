#!/usr/bin/env bash

SCRIPT_PATH=$(dirname "$BASH_SOURCE")

# Global variables to make the parsing of input arguments a bit easier
INPUT_FILES=()
OPTIONS=()
OUTPUT_FILE=""
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
        OPTIONS+=($1)

        if [[ $1 == "-c" ]]; then
          COMPILE_ONLY="True"
          SHOULD_LINK="False"
        elif [[ $1 == "-init-only" ]]; then
          SHOULD_LINK="False"
        elif [[ $1 == "-emit-flattened" ]]; then
          SHOULD_LINK="False"
        elif [[ $1 == "-emit-ast" ]]; then
          SHOULD_LINK="False"
        elif [[ $1 == "-emit-modelica-dialect" ]]; then
          SHOULD_LINK="False"
        elif [[ $1 == "-emit-llvm-dialect" ]]; then
          SHOULD_LINK="False"
        elif [[ $1 == "-emit-llvm-ir" ]]; then
          SHOULD_LINK="False"
        fi

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

main() {
  parse_args "$@"
  temporary_dir=$(mktemp -d)

  marco-driver "${OPTIONS[@]}" "${INPUT_FILES[@]}" -o "$temporary_dir/simulation.o"
  resultCode=$?

  if [ $resultCode -ne 0 ]; then
    echo "Compilation error"
    rm -rf "${temporary_dir}"
    exit $resultCode
  fi

  if [[ $COMPILE_ONLY == "True" ]]; then
    mv "$temporary_dir/simulation.o" "$OUTPUT_FILE.o"
    rm -rf "${temporary_dir}"
    exit 0
  fi

  if [[ $SHOULD_LINK == "True" ]]; then
    g++ "$temporary_dir/simulation.o" $SCRIPT_PATH/../lib/libMARCORuntime.so -o "$OUTPUT_FILE" -Wl,-R$SCRIPT_PATH/../lib
    resultCode=$?

    if [ $resultCode -ne 0 ]; then
      echo "Link error"
      rm -rf "${temporary_dir}"
      exit $resultCode
    fi
  fi

  rm -r $temporary_dir
}

main "${@}"
