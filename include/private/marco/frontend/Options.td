//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by MARCO.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

/////////
// Flags

// MC1Option - This option should be accepted by clang -cc1.
def MC1Option : OptionFlag;

// A short name to show in documentation.
class DocName<string name> { string DocName = name; }

// A brief description to show in documentation.
class DocBrief<code descr> { code DocBrief = descr; }

// Indicates that this group should be flattened into its parent when generating
// documentation.
class DocFlatten { bit DocFlatten = 1; }

/////////
// Groups

def Action_Group : OptionGroup<"<action group>">,
    DocName<"Actions">,
    DocBrief<[{The action to perform on the input.}]>;

def DebugInfo_Group : OptionGroup<"<g group>">,
    DocName<"Debug information generation">,
    DocBrief<[{Flags controlling how much and what kind of debug information should be generated.}]>;

def g_Group : OptionGroup<"<g group>">,
    Group<DebugInfo_Group>,
    DocName<"Kind and level of debug information">;

def OpenModelica_Group : OptionGroup<"<omc group>">,
    DocName<"OpenModelica compiler">,
    DocBrief<[{Flags controlling the OpenModelica compiler.}]>;

def omc_Group : OptionGroup<"<omc group>">,
    Group<OpenModelica_Group>,
    DocName<"OpenModelica compiler">;

def Codegen_Group : OptionGroup<"<c group>">,
    DocName<"Code generation">,
    DocBrief<[{Flags controlling the code generation and lowering processes.}]>;

def c_Group : OptionGroup<"<c group>">,
    Group<Codegen_Group>,
    DocName<"Code generation and lowering">;

def Simulation_Group : OptionGroup<"<s group>">,
    DocName<"Simulation">,
    DocBrief<[{Flags controlling the simulation options.}]>;

def s_Group : OptionGroup<"<s group>">,
    Group<Simulation_Group>,
    DocName<"Simulation options">;

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

// TableGen record for a single marshalled flag.
class MarshalledFlagRec<FlagDefExpanded flag, FlagDefExpanded other,
                        FlagDefExpanded implied, KeyPathAndMacro kpm,
                        Default default>
  : Flag<["-"], flag.Spelling>, Flags<flag.OptionFlags>, HelpText<flag.Help>,
    MarshallingInfoBooleanFlag<kpm, default.Value, flag.ValueAsCode,
                               flag.RecordName, other.ValueAsCode,
                               other.RecordName>,
    ImpliedByAnyOf<implied.ImpliedBy, implied.ValueAsCode> {}

// Generates TableGen records for two command line flags that control the same
// key path via the marshalling infrastructure.
// Names of the records consist of the specified prefix, "no_" for the negative
// flag, and NAME.
// Used for -cc1 frontend options. Driver-only options do not map to
// CompilerInvocation.
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

/////////
// Options

def help : Flag<["-", "--"], "help">,
    Flags<[MC1Option]>,
    HelpText<"Display available options">,
    MarshallingInfoFlag<FrontendOpts<"ShowHelp">>;

def version : Flag<["-", "--"], "version">,
    Flags<[MC1Option]>,
    HelpText<"Print the compiler version">,
    MarshallingInfoFlag<FrontendOpts<"ShowVersion">>;

def o : JoinedOrSeparate<["-"], "o">,
    Flags<[MC1Option]>,
    HelpText<"Write output to <file>">, MetaVarName<"<file>">,
    MarshallingInfoString<FrontendOpts<"OutputFile">>;

def debug : Flag<["-"], "g">,
    Group<g_Group>,
    HelpText<"Generate source-level debug information">;

def omc_path : Joined<["--"], "omc-path=">,
    Flags<[MC1Option]>,
    HelpText<"Set a custom path for the OpenModelica compiler">;

def omc_arg : Joined<["--"], "Xomc=">,
    Flags<[MC1Option]>,
    HelpText<"Pass a custom argument to the OpenModelica compiler (the option can be used multiple times)">;

def omc_bypass : Flag<["--"], "omc-bypass">,
    Flags<[MC1Option]>,
    HelpText<"Bypass OMC and pass the source directly to MARCO">;

defm assertions : BoolOption<"", "assertions",
    CodegenOpts<"Assertions">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Generate">,
    NegFlag<SetFalse, [], "Don't generate">,
    BothFlags<[MC1Option], " assertions">>,
    Group<c_Group>;

defm generate_main : BoolOption<"", "generate-main",
    CodegenOpts<"GenerateMain">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Generate">,
    NegFlag<SetFalse, [], "Don't generate">,
    BothFlags<[MC1Option], " the main function that starts and manages the simulation">>,
    Group<c_Group>;

defm function_inlining : BoolOption<"", "function-inlining",
    CodegenOpts<"FunctionInlining">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Inline functions when suggested by the annotations">,
    NegFlag<SetFalse, [], "Never inline functions">,
    BothFlags<[MC1Option], "">>,
    Group<c_Group>;

defm output_arrays_promotion : BoolOption<"", "output-arrays-promotion",
    CodegenOpts<"OutputArraysPromotion">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Convert">,
    NegFlag<SetFalse, [], "Don't convert">,
    BothFlags<[MC1Option], " output arrays into arguments to delegate the allocation to the caller">>,
    Group<c_Group>;

defm cse : BoolOption<"", "cse",
    CodegenOpts<"CSE">,
    DefaultTrue,
    PosFlag<SetTrue, [], "Enable">,
    NegFlag<SetFalse, [], "Disable">,
    BothFlags<[MC1Option], " common subexpression elimination">>,
    Group<c_Group>;

defm omp : BoolOption<"", "omp",
    CodegenOpts<"OpenMP">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Enable OpenMP usage (experimental)">,
    NegFlag<SetFalse, [], "Disable OpenMP usage">,
    BothFlags<[MC1Option], "">>,
    Group<c_Group>;

defm c_wrappers : BoolOption<"", "c-wrappers",
    CodegenOpts<"CWrappers">,
    DefaultFalse,
    PosFlag<SetTrue, [], "Generate">,
    NegFlag<SetFalse, [], "Don't generate">,
    BothFlags<[MC1Option], " the C wrappers">>,
    Group<c_Group>;

def opt : Joined<["-"], "O">,
  Flags<[MC1Option]>,
  HelpText<"Set optimization level">,
  Group<c_Group>;

def : Alias<opt>, AliasArgs<["0"]>,
  HelpText<"Disable optimization">;

def : Alias<opt>, AliasArgs<["1"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["2"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["3"]>,
  HelpText<"Optimize for speed">;

def : Alias<opt>, AliasArgs<["s"]>,
  HelpText<"Optimize for size">;

def model : Joined<["--"], "model=">,
    Group<s_Group>,
    Flags<[MC1Option]>,
    HelpText<"Model name">;

def start_time : Joined<["--"], "start-time=">,
    Group<s_Group>,
    Flags<[MC1Option]>,
    HelpText<"Start time of the simulation (in seconds)">;

def end_time : Joined<["--"], "end-time=">,
    Group<s_Group>,
    Flags<[MC1Option]>,
    HelpText<"End time of the simulation (in seconds)">;

def time_step : Joined<["--"], "time-step=">,
    Group<s_Group>,
    Flags<[MC1Option]>,
    HelpText<"Time step of the simulation (in seconds)">;

//===----------------------------------------------------------------------===//
// Frontend Options - mc1
//===----------------------------------------------------------------------===//

def init_only : Flag<["--"], "init-only">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Only execute frontend initialization">;

def emit_flattened : Flag<["--"], "emit-flattened">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit flattened Modelica">;

def emit_ast : Flag<["--"], "emit-ast">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit AST">;

def emit_modelica_dialect : Flag<["--"], "emit-modelica-dialect">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit Modelica dialect">;

def emit_llvm_dialect : Flag<["--"], "emit-llvm-dialect">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM dialect">;

def emit_llvm_ir : Flag<["--"], "emit-llvm-ir">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM-IR">;

// Used by the driver script to determine whether the link phase should take place.
def compile_only : Flag<["-"], "c">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Compile only">;
