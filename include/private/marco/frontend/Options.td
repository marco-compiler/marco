//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by MARCO.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

// MC1Option - This option should be accepted by clang -cc1.
def MC1Option : OptionFlag;

class FrontendOpts<string base>
    : KeyPathAndMacro<"FrontendOpts.", base, "FRONTEND_"> {}

def help : Flag<["-", "--"], "help">,
    Flags<[MC1Option]>,
    HelpText<"Display available options">,
    MarshallingInfoFlag<FrontendOpts<"ShowHelp">>;

def version : Flag<["-"], "version">,
    Flags<[MC1Option]>,
    HelpText<"Print the compiler version">,
    MarshallingInfoFlag<FrontendOpts<"ShowVersion">>;
