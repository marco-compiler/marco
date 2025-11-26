#ifndef MARCO_FRONTEND_PASSES_TD
#define MARCO_FRONTEND_PASSES_TD

include "mlir/Pass/PassBase.td"

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
