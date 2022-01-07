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

class FrontendOpts<string base>
    : KeyPathAndMacro<"FrontendOpts.", base, "FRONTEND_"> {}

/////////
// Options

def help : Flag<["-", "--"], "help">,
    Flags<[MC1Option]>,
    HelpText<"Display available options">,
    MarshallingInfoFlag<FrontendOpts<"ShowHelp">>;

def version : Flag<["-"], "version">,
    Flags<[MC1Option]>,
    HelpText<"Print the compiler version">,
    MarshallingInfoFlag<FrontendOpts<"ShowVersion">>;

def o : JoinedOrSeparate<["-"], "o">,
    Flags<[MC1Option]>,
    HelpText<"Write output to <file>">, MetaVarName<"<file>">,
    MarshallingInfoString<FrontendOpts<"OutputFile">>;

def x : JoinedOrSeparate<["-"], "x">,
    Flags<[MC1Option]>,
    HelpText<"Treat subsequent input files as having type <language>">,
    MetaVarName<"<language>">;

def g_Flag : Flag<["-"], "g">,
    Group<g_Group>,
    HelpText<"Generate source-level debug information">;

//===----------------------------------------------------------------------===//
// Frontend Options - mc1
//===----------------------------------------------------------------------===//

def init_only : Flag<["-", "--"], "init-only">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Only execute frontend initialization">;

def emit_ast : Flag<["-", "--"], "emit-ast">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit AST">;

def emit_modelica_dialect : Flag<["-", "--"], "emit-modelica-dialect">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit Modelica dialect">;

def emit_llvm_dialect : Flag<["-", "--"], "emit-llvm-dialect">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM dialect">;

def emit_llvm_ir : Flag<["-", "--"], "emit-llvm-ir">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit LLVM-IR">;

def emit_bitcode : Flag<["-", "--"], "emit-bitcode">,
    Flags<[MC1Option]>,
    Group<Action_Group>,
    HelpText<"Emit bitcode">;
