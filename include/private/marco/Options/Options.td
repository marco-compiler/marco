//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by MARCO.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

// A short name to show in documentation.
class DocName<string name> { string DocName = name; }

// A brief description to show in documentation.
class DocBrief<code descr> { code DocBrief = descr; }

// Indicates that this group should be flattened into its parent when generating
// documentation.
class DocFlatten { bit DocFlatten = 1; }

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

// DriverOption - This option should be accepted by marco.
def DriverOption : OptionFlag;

// MC1Option - This option should be accepted by marco -mc1.
def MC1Option : OptionFlag;

//===----------------------------------------------------------------------===//
// Groups
//===----------------------------------------------------------------------===//

def Action_Group : OptionGroup<"Actions">,
    DocName<"Actions">,
    DocBrief<[{The action to perform on the input.}]>;

def DebugInfo_Group : OptionGroup<"Debug info">,
    DocName<"Debug information generation">,
    DocBrief<[{Flags controlling how much and what kind of debug information should be generated.}]>;

def OpenModelica_Group : OptionGroup<"OMC">,
    DocName<"OpenModelica compiler">,
    DocBrief<[{Flags controlling the OpenModelica compiler.}]>;

def IDA_Group : OptionGroup<"IDA">,
    DocName<"IDA">,
    DocBrief<[{Flags controlling the IDA solver.}]>;

def Codegen_Group : OptionGroup<"Code generation">,
    DocName<"Code generation">,
    DocBrief<[{Flags controlling the code generation and lowering processes.}]>;

def Simulation_Group : OptionGroup<"Simulation">,
    DocName<"Simulation">,
    DocBrief<[{Flags controlling the simulation options.}]>;

def Link_Group : OptionGroup<"Linking">,
    DocName<"Linking">,
    DocBrief<[{Flags controlling the linking process.}]>;

class FrontendOpts<string base>
    : KeyPathAndMacro<"FrontendOpts.", base, "FRONTEND_"> {}

class CodegenOpts<string base>
    : KeyPathAndMacro<"CodegenOpts.", base, "CODEGEN_"> {}

//===----------------------------------------------------------------------===//
// BoolOption
//===----------------------------------------------------------------------===//

// The default value of a marshalled key path.
class Default<code value> { code Value = value; }

// Convenience variables for boolean defaults.
def DefaultTrue : Default<"true"> {}
def DefaultFalse : Default<"false"> {}

// The value set to the key path when the flag is present on the command line.
class Set<bit value> { bit Value = value; }
def SetTrue : Set<true> {}
def SetFalse : Set<false> {}

// Definition of single command line flag. This is an implementation detail, use
// SetTrueBy or SetFalseBy instead.
class FlagDef<bit polarity, bit value, list<OptionFlag> option_flags,
              string help, list<code> implied_by_expressions = []> {
  // The polarity. Besides spelling, this also decides whether the TableGen
  // record will be prefixed with "no_".
  bit Polarity = polarity;

  // The value assigned to key path when the flag is present on command line.
  bit Value = value;

  // OptionFlags that control visibility of the flag in different tools.
  list<OptionFlag> OptionFlags = option_flags;

  // The help text associated with the flag.
  string Help = help;

  // List of expressions that, when true, imply this flag.
  list<code> ImpliedBy = implied_by_expressions;
}

// Additional information to be appended to both positive and negative flag.
class BothFlags<list<OptionFlag> option_flags, string help = ""> {
  list<OptionFlag> OptionFlags = option_flags;
  string Help = help;
}

// Functor that appends the suffix to the base flag definition.
class ApplySuffix<FlagDef flag, BothFlags suffix> {
  FlagDef Result
    = FlagDef<flag.Polarity, flag.Value,
              flag.OptionFlags # suffix.OptionFlags,
              flag.Help # suffix.Help, flag.ImpliedBy>;
}

// Definition of the command line flag with positive spelling, e.g. "-ffoo".
class PosFlag<Set value, list<OptionFlag> flags = [], string help = "",
              list<code> implied_by_expressions = []>
  : FlagDef<true, value.Value, flags, help, implied_by_expressions> {}

// Definition of the command line flag with negative spelling, e.g. "-fno-foo".
class NegFlag<Set value, list<OptionFlag> flags = [], string help = "",
              list<code> implied_by_expressions = []>
  : FlagDef<false, value.Value, flags, help, implied_by_expressions> {}

// Expanded FlagDef that's convenient for creation of TableGen records.
class FlagDefExpanded<FlagDef flag, string prefix, string name, string spelling>
  : FlagDef<flag.Polarity, flag.Value, flag.OptionFlags, flag.Help,
            flag.ImpliedBy> {
  // Name of the TableGen record.
  string RecordName = prefix # !if(flag.Polarity, "", "no_") # name;

  // Spelling of the flag.
  string Spelling = prefix # !if(flag.Polarity, "", "no-") # spelling;

  // Can the flag be implied by another flag?
  bit CanBeImplied = !not(!empty(flag.ImpliedBy));

  // C++ code that will be assigned to the keypath when the flag is present.
  code ValueAsCode = !if(flag.Value, "true", "false");
}

class MarshalledFlagRec<FlagDefExpanded flag, FlagDefExpanded other,
                        FlagDefExpanded implied, KeyPathAndMacro kpm,
                        Default default>
  : Flag<["--"], flag.Spelling>, Flags<flag.OptionFlags>, HelpText<flag.Help>,
    MarshallingInfoBooleanFlag<kpm, default.Value, flag.ValueAsCode,
                               other.ValueAsCode, other.RecordName>,
    ImpliedByAnyOf<implied.ImpliedBy, implied.ValueAsCode> {}

// Generates TableGen records for two command line flags that control the same
// key path via the marshalling infrastructure.
// Names of the records consist of the specified prefix, "no_" for the negative
// flag, and NAME.
multiclass BoolOption<string prefix = "", string spelling_base,
                      KeyPathAndMacro kpm, Default default,
                      FlagDef flag1_base, FlagDef flag2_base,
                      BothFlags suffix = BothFlags<[], "">> {
  defvar flag1 = FlagDefExpanded<ApplySuffix<flag1_base, suffix>.Result, prefix,
                                 NAME, spelling_base>;

  defvar flag2 = FlagDefExpanded<ApplySuffix<flag2_base, suffix>.Result, prefix,
                                 NAME, spelling_base>;

  // The flags must have different polarity, different values, and only
  // one can be implied.
  assert !xor(flag1.Polarity, flag2.Polarity),
         "the flags must have different polarity: flag1: " #
             flag1.Polarity # ", flag2: " # flag2.Polarity;
  assert !ne(flag1.Value, flag2.Value),
         "the flags must have different values: flag1: " #
             flag1.Value # ", flag2: " # flag2.Value;
  assert !not(!and(flag1.CanBeImplied, flag2.CanBeImplied)),
         "only one of the flags can be implied: flag1: " #
             flag1.CanBeImplied # ", flag2: " # flag2.CanBeImplied;

  defvar implied = !if(flag1.CanBeImplied, flag1, flag2);

  def flag1.RecordName : MarshalledFlagRec<flag1, flag2, implied, kpm, default>;
  def flag2.RecordName : MarshalledFlagRec<flag2, flag1, implied, kpm, default>;
}

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

def help : Flag<["--"], "help">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Display available options">,
    MarshallingInfoFlag<FrontendOpts<"ShowHelp">>;

def version : Flag<["--"], "version">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Print the compiler version">,
    MarshallingInfoFlag<FrontendOpts<"ShowVersion">>;

def print_statistics : Flag<["--"], "print-statistics">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Print statistics">;

defm multithreading : BoolOption<"", "multithreading",
    CodegenOpts<"Multithreading">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Use">,
    NegFlag<SetFalse, [], "Don't use">,
    BothFlags<[DriverOption, MC1Option], " multithreading">>,
    Group<Codegen_Group>;

def o : JoinedOrSeparate<["-"], "o">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Write output to <file>">,
    MetaVarName<"<file>">,
    MarshallingInfoString<FrontendOpts<"OutputFile">>;

def debug : Flag<["-"], "g">,
    Flags<[DriverOption, MC1Option]>,
    Group<DebugInfo_Group>,
    HelpText<"Generate source-level debug information">;

def omc_path : Joined<["--"], "omc-path=">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Set a custom path for the OpenModelica compiler">;

def omc_arg : Joined<["--"], "Xomc=">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Pass a custom argument to the OpenModelica compiler (the option can be used multiple times)">;

def omc_bypass : Flag<["--"], "omc-bypass">,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Bypass OMC and pass the source directly to MARCO">;

defm assertions : BoolOption<"", "assertions",
    CodegenOpts<"Assertions">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Generate">,
    NegFlag<SetFalse, [], "Don't generate">,
    BothFlags<[DriverOption, MC1Option], " assertions">>,
    Group<Codegen_Group>;

defm generate_main : BoolOption<"", "generate-main",
    CodegenOpts<"GenerateMain">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Generate">,
    NegFlag<SetFalse, [], "Don't generate">,
    BothFlags<[DriverOption], " the main function that starts the simulation">>,
    Group<Codegen_Group>;

def bit_width : Joined<["--"], "bit-width=">,
    Flags<[DriverOption, MC1Option]>,
    Group<Codegen_Group>,
    HelpText<"Set the bit-width of the simulation data">;

defm function_inlining : BoolOption<"", "inlining",
    CodegenOpts<"Inlining">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Inline functions when suggested by the annotations">,
    NegFlag<SetFalse, [], "Never inline functions">,
    BothFlags<[DriverOption, MC1Option], "">>,
    Group<Codegen_Group>;

defm output_arrays_promotion : BoolOption<"", "output-arrays-promotion",
    CodegenOpts<"OutputArraysPromotion">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Convert">,
    NegFlag<SetFalse, [], "Don't convert">,
    BothFlags<[DriverOption, MC1Option], " output arrays into arguments to delegate the allocation to the caller">>,
    Group<Codegen_Group>;

defm read_only_variables_propagation : BoolOption<"", "read-only-variables-propagation",
    CodegenOpts<"ParametersPropagation">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Propagate">,
    NegFlag<SetFalse, [], "Don't propagate">,
    BothFlags<[DriverOption, MC1Option], " the values assigned to read-only variables through binding equations">>,
    Group<Codegen_Group>;

defm variables_to_parameters_promotion : BoolOption<"", "variables-to-parameters-promotion",
    CodegenOpts<"VariablesToParametersPromotion">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Promote">,
    NegFlag<SetFalse, [], "Don't promote">,
    BothFlags<[DriverOption, MC1Option], " variables to parameters when depending only on constants or parameters">>,
    Group<Codegen_Group>;

defm cse : BoolOption<"", "cse",
    CodegenOpts<"CSE">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Enable">,
    NegFlag<SetFalse, [], "Disable">,
    BothFlags<[DriverOption, MC1Option], " common subexpression elimination">>,
    Group<Codegen_Group>;

defm omp : BoolOption<"", "omp",
    CodegenOpts<"OpenMP">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Enable OpenMP usage (experimental)">,
    NegFlag<SetFalse, [], "Disable OpenMP usage">,
    BothFlags<[DriverOption, MC1Option], "">>,
    Group<Codegen_Group>;

def opt : Joined<["-"], "O">,
  Flags<[DriverOption, MC1Option]>,
  HelpText<"Set optimization level">,
  Group<Codegen_Group>;

def : Alias<opt>, AliasArgs<["0"]>,
  HelpText<"Disable optimizations">;

def : Alias<opt>, AliasArgs<["1"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["2"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["3"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["s"]>,
  HelpText<"Optimize for size">;

def : Alias<opt>, AliasArgs<["z"]>,
  HelpText<"Optimize for size">;

def target : Joined<["--"], "target=">,
    Flags<[DriverOption, MC1Option]>,
    Group<Codegen_Group>,
    HelpText<"Generate code for the given target">;

def cpu: Joined<["--"], "mcpu=">,
    Flags<[DriverOption, MC1Option]>,
    Group<Codegen_Group>,
    HelpText<"Optimize code for a given CPU">;

def model : Joined<["--"], "model=">,
    Group<Simulation_Group>,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Model name">;

def filter : Joined<["--"], "variable-filter=">,
    Group<Simulation_Group>,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Filter for the variables to be printed">;

def solver : Joined<["--"], "solver=">,
    Group<Simulation_Group>,
    Flags<[DriverOption, MC1Option]>,
    HelpText<"Solver (possible options: euler-forward, ida)">;

defm ida_reduced_system : BoolOption<"", "ida-reduced-system",
    CodegenOpts<"IDA">,
    DefaultTrue,
    PosFlag<SetTrue, [], "IDA: compute a reduced system for the external solver">,
    NegFlag<SetFalse, [], "IDA: all variables and equations are passed to the external solver">,
    BothFlags<[DriverOption, MC1Option], "">>,
    Group<Simulation_Group>;

defm ida_reduced_derivatives : BoolOption<"", "ida-reduced-derivatives",
    CodegenOpts<"IDA">,
    DefaultTrue,
    PosFlag<SetTrue, [], "IDA: determine the derivatives that are always equal to zero">,
    NegFlag<SetFalse, [], "IDA: do not determine the derivatives that are always equal to zero">,
    BothFlags<[DriverOption, MC1Option], "">>,
    Group<Simulation_Group>;

defm ida_jacobian_one_sweep : BoolOption<"", "ida-jacobian-one-sweep",
    CodegenOpts<"IDA">,
    DefaultTrue,
    PosFlag<SetTrue, [], "IDA: enable one-sweep computation of derivatives with respect to state and derivative variables">,
    NegFlag<SetFalse, [], "IDA: disable one-sweep computation of derivatives with respect to state and derivative variables">,
    BothFlags<[DriverOption, MC1Option], "">>,
    Group<Simulation_Group>;

//===----------------------------------------------------------------------===//
// Frontend Options - mc1
//===----------------------------------------------------------------------===//

def init_only : Flag<["--"], "init-only">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Only execute frontend initialization">;

def emit_flattened : Flag<["--"], "emit-flattened">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit flattened Modelica">;

def emit_ast : Flag<["--"], "emit-ast">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit AST">;

def emit_final_ast : Flag<["--"], "emit-final-ast">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit the final AST, after the frontend passes">;

def emit_mlir : Flag<["--"], "emit-mlir">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit MLIR">;

def emit_llvm_ir : Flag<["--"], "emit-llvm-ir">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM-IR">;

def emit_llvm_bitcode : Flag<["--"], "emit-llvm-bc">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM bitcode">;

def compile_only : Flag<["-"], "S">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Compile only; do not assemble or link">;

def compile_and_assemble_only : Flag<["-"], "c">,
    Flags<[DriverOption, MC1Option]>,
    Group<Action_Group>,
    HelpText<"Compile and assemble, but do not link">;

def linker_arg : Separate<["-"], "Xlinker">,
    Flags<[DriverOption]>,
    HelpText<"Pass <arg> to the linker">;
