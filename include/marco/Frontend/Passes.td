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

def EquationIndexCheckInsertionPass
    : Pass<"equation-index-check-insertion", "mlir::func::FuncOp">
{
    let summary = "Insert bound checks for equation indices.";

    let description = [{
        Insert bound checks for equation indices to ensure they are within
        valid ranges during runtime. This pass is used when offloading
        equations to GPUs to ensure that threads do not access out-of-bounds
        memory.
    }];

    let dependentDialects = [
        "mlir::affine::AffineDialect"
    ];

    let constructor = "marco::frontend::createEquationIndexCheckInsertionPass()";
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
