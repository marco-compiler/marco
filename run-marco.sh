#!/usr/bin/env bash

#trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
SCRIPTPATH=$(dirname "$BASH_SOURCE")

LOG=/dev/null

marco_find_external()
{
  # $1: executable to find
  
  path=$(which "$1")
  if [[ -e "$path" ]]; then
    echo "$path"
    return 0
  else
    echo "Cannot find $1" >> $LOG >> /dev/stderr 
    return 1
  fi
  
}

marco_find_internal()
{
  # $1: prefix
  # $2: directory in prefix
  # $3: file to find
  
  cd $1
  BASEDIR=$(pwd)

  if [[ $2 == lib* ]]; then
    SOEXT="so"
    if [ $(uname -s) = "Darwin" ]; then
      SOEXT="dylib";
    fi
    FN="$3.$SOEXT"
  else
    FN="$3"
  fi

  PATH="$BASEDIR/$2/$FN"

  if [[ ! -e "$PATH" ]]; then
    echo "Cannot find $FN, $PATH does not exist" >> $LOG >> /dev/stderr
  else
    echo "Found $PATH" >> $LOG
    echo "$PATH"
  fi
}

MARCO_PREFIX="$SCRIPTPATH/build"

OMC=$(marco_find_external 'omc')
MARCO=$(marco_find_internal "$MARCO_PREFIX" 'bin' 'marco')
MARCO_RT=$(marco_find_internal "$MARCO_PREFIX" 'lib/runtime' 'libMARCORuntime')

if [[ -z "$LLVM_DIR" ]]; then
  LLVM_DIR=$(llvm-config --prefix 2> /dev/null)
  if [[ $? -ne 0 ]]; then
    printf "*** WARNING ***\nCannot set LLVM_DIR using llvm-config\n"
  fi
fi
if [[ ! ( -z "$LLVM_DIR" ) ]]; then
  if [ $(uname -s) = "Darwin" ]; then
    # xcrun patches the command line parameters to clang to add the standard
    # include paths depending on where the currently active platform SDK is.
    # This is necessary only if you build clang yourself, distributions of
    # clang from major package managers and Apple already do this automatically
    export CLANG="xcrun $LLVM_DIR/bin/clang"
    export CLANGXX="xcrun $LLVM_DIR/bin/clang++"
  fi
fi

if [[ -z $LLVM_DIR ]]; then
  echo -e '\033[33m'"Warning"'\033[39m'" using default llvm/clang";
else
  llvmbin="$LLVM_DIR/bin/";
fi
if [[ -z "$CLANG" ]]; then CLANG=${llvmbin}clang; fi
if [[ -z "$CLANGXX" ]]; then CLANGXX=${CLANG}++; fi
if [[ -z "$OPT" ]]; then OPT=${llvmbin}opt; fi
if [[ -z "$LLC" ]]; then LLC=${llvmbin}llc; fi
if [[ -z "$LLVM_LINK" ]]; then LLVM_LINK=${llvmbin}llvm-link; fi

llvm_debug=$(($("$OPT" --version | grep DEBUG | wc -l)))

# option parser state
raw_opts="$@"
parse_state=INITIAL
help=0
# global options
optimization=
temporary_dir=$(mktemp -d)
del_temporary_dir=1
# OMC flattening configs
input_files=()
instantiate=
omc_debug='-d=newInst,nonfScalarize'
omc_opts=
# MARCO frontend configs
marco_opts=
# CLANG compiler configs
output_file="a.out"
clang_opts=

for opt in $raw_opts; do
  case $parse_state in
    INITIAL)
      case $opt in
        -o*)
          if [[ ${#opt} -eq 2 ]]; then
            parse_state=OUT_FILE;
          else
            output_file=`echo "$opt" | cut -b 2`;
          fi;
          ;;
        -O*)
          optimization=$opt;
          ;;
        -i | --instClass)
          parse_state=INSTANTIATE
          ;;
        -i*)
          instantiate="$instantiate $opt"
          ;;
        -d | --debug)
          parse_state=OMC_DEBUG
          ;;
        -d*)
          omc_debug="$omc_debug $opt"
          ;;
        -enable-log)
          LOG=/dev/stderr;
          ;;
        -temp-dir)
          del_temporary_dir=0
          parse_state=TEMP_DIR
          ;;
        -help | -h | -version | -v | --help | --version)
          help=1
          ;;
        -Xomc)
          parse_state=OMC_OPT
          ;;
        -Xmarco)
          parse_state=MARCO_OPT
          ;;
        -Xclang)
          parse_state=CLANG_OPT
          ;;
        -*)
          marco_opts="$opts $opt";
          ;;
        *.mo)
          input_files+=( "$opt" );
          ;;
        *)
          marco_opts="$opts $opt";
          ;;
      esac;
      ;;
    OUT_FILE)
      output_file="$opt";
      parse_state=INITIAL;
      ;;
    TEMP_DIR)
      temporary_dir="$opt";
      parse_state=INITIAL;
      ;;
    INSTANTIATE)
      instantiate="$instantiate -i=$opt"
      parse_state=INITIAL;
      ;;
    OMC_DEBUG)
      omc_debug="$omc_debug -d=$opt"
      parse_state=INITIAL;
      ;;
    OMC_OPT)
      omc_opts="$omc_opts $opt"
      parse_state=INITIAL;
      ;;
    MARCO_OPT)
      marco_opts="$marco_opts $opt"
      parse_state=INITIAL;
      ;;
    CLANG_OPT)
      clang_opts="$clang_opts $opt"
      parse_state=INITIAL;
      ;;
  esac;
done

output_basename=$(basename ${output_file})

if [[ ( -z "$input_files" ) || ( $help -ne 0 ) ]]; then
  cat << HELP_END
RUN-MARCO: Driver for MARCO, the Modelica Advanced Research COmpiler
Usage: run-marco.sh [options] file...

The specified file can be any Modelica MO source file. MOS scripts are NOT
accepted. This script also accepts any option that can be passed to the
MARCO frontend.

Options:
  -o <file>             Write compilation output to <file>
  -O<level>             Set the optimization level to the specified value.
                        The accepted optimization levels are the same as CLANG.
                        (-O, -O1, -O2, -O3, -Os, -Of)
  -enable-log           Enable driver debug logging during the compilation.
  -temp-dir <dir>       Store various temporary files related to the execution
                        of MARCO to the specified directory.
  -i, --instClass <cls> Instantiate the class given by the fully qualified
                        path
  -d, --debug <opts>    Add OpenModelica debug options
  --end-time=<number>   End time (in seconds) (default: 10)
  --start-time=<number> Start time (in seconds) (default: 0)
  --time-step=<number>  Time step (in seconds) (default: 0.1)
  -Xomc <arg>           Pass <arg> to OMC when flattening
  -Xmarco <arg>         Pass <arg> to the MARCO frontend
  -Xclang <arg>         Pass <arg> to clang when linking
HELP_END
  exit 0
fi

# enable bash logging
if [[ $LOG != /dev/null ]]; then
  set -x
fi

###
### FLATTENING
###
${OMC} \
  $instantiate \
  $omc_debug \
  $omc_opts \
  "${input_files[@]}" \
    > "${temporary_dir}/flattened.mo" || exit $?
    
###
### FRONTEND
###
${MARCO} \
  $optimization \
  $marco_opts \
  "${temporary_dir}/flattened.mo" \
  -o "${temporary_dir}/simulation.bc" || exit $?

###
### BACKEND
###
if [ $(uname -s) = "Darwin" ]; then
  clang_opts="$clang_opts -rpath $(dirname $MARCO_RT)"
fi
${CLANG} \
  "${temporary_dir}/simulation.bc" \
  -L "$(dirname $MARCO_RT)" -lMARCORuntime \
  $optimization \
  $clang_opts \
  -o "$output_file" || exit $?

###
###  Cleanup
###
if [[ $del_temporary_dir -ne 0 ]]; then
  rm -rf "${temporary_dir}"
fi
