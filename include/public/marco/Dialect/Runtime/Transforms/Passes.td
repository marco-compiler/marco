#ifndef MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_TD
#define MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

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

    let constructor = "mlir::runtime::createHeapFunctionsReplacementPass()";
}

#endif // MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_TD
