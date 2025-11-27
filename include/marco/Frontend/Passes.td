#ifndef MARCO_FRONTEND_PASSES_TD
#define MARCO_FRONTEND_PASSES_TD

include "mlir/Pass/PassBase.td"

def AggressiveLICMPass : Pass<"aggressive-licm">
{
    let summary = "Move loop-invariant code out of loop bodies.";

    let description = [{
        Perform LICM in a more aggressive way if compared to the upstream pass.
        In particular, it allows hoisting operations that may have side-effects
        (e.g., memory writes) as long as it can prove that such side-effects do
        not go beyond the scope of the module (i.e., there are not write
        effects on MLIR's DefaultResource resource).
    }];

    let constructor = "marco::frontend::createAggressiveLICMPass()";
}

def HeapFunctionsReplacementPass
    : Pass<"heap-functions-replacement", "mlir::ModuleOp">
{
    let summary = "Replace heap management functions with custom ones.";

    let description = [{
        Replace heap management functions (malloc, free) with custom ones
        living inside the runtime library.
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
    ];

    let constructor = "marco::frontend::createHeapFunctionsReplacementPass()";
}

#endif // MARCO_FRONTEND_PASSES_TD
