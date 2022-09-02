#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_TD
#define MARCO_CODEGEN_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ArrayDeallocationPass : Pass<"array-deallocation", "mlir::ModuleOp"> {
    let summary = "Automatic array deallocation";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createArrayDeallocationPass()";
}

def AutomaticDifferentiationPass : Pass<"auto-diff", "mlir::ModuleOp"> {
    let summary = "Automatic differentiation of functions";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createAutomaticDifferentiationPass()";
}

def ExplicitCastInsertionPass : Pass<"explicit-cast-insertion", "mlir::ModuleOp"> {
    let summary = "Explicit cast insertion";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::modelica::createExplicitCastInsertionPass()";
}

def FunctionScalarizationPass : Pass<"scalarize", "mlir::ModuleOp"> {
    let summary = "Scalarize";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithmeticDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::scf::SCFDialect"
    ];

    let options = [
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
    ];

  let constructor = "mlir::modelica::createFunctionScalarizationPass()";
}

def ModelSolvingPass : Pass<"solve-model", "mlir::ModuleOp"> {
    let summary = "Solve the model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithmeticDialect",
        "mlir::func::FuncDialect",
        "mlir::ida::IDADialect",
        "mlir::modelica::ModelicaDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"startTime", "start-time", "double", "0",
               "Set the simulation start time">,
        Option<"endTime", "end-time", "double", "1",
               "Set the simulation end time">,
        Option<"timeStep", "time-step", "double", "0.1",
               "Set the simulation time step">,
        Option<"emitMain", "emit-main", "bool", "true",
               "Generate the main function of the simulation">,
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">
    ];

    let constructor = "mlir::modelica::createModelSolvingPass()";
}

def MatchingTestPass : Pass<"matching", "mlir::ModuleOp"> {
    let summary = "Perform the matching on the model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithmeticDialect"
    ];

    let constructor = "mlir::modelica::createMatchingTestPass()";
}

def SchedulingTestPass : Pass<"scheduling", "mlir::ModuleOp"> {
    let summary = "Perform the scheduling on the model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithmeticDialect"
    ];

    let constructor = "mlir::modelica::createSchedulingTestPass()";
}

def NegateOpDistributionPass : Pass<"distribute-neg", "mlir::ModuleOp"> {
    let summary = "Distribute the negation operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createNegateOpDistributionPass()";
}

def MulOpDistributionPass : Pass<"distribute-mul", "mlir::ModuleOp"> {
    let summary = "Distribute the multiplication operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createMulOpDistributionPass()";
}

def DivOpDistributionPass : Pass<"distribute-div", "mlir::ModuleOp"> {
    let summary = "Distribute the division operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createDivOpDistributionPass()";
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_TD
