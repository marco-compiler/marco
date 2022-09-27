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

def ModelLegalizationPass : Pass<"legalize-model", "mlir::ModuleOp"> {
    let summary = "Process a model in order to make it solvable";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithmeticDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model to be processed">,
        Option<"debugView", "debug-view", "bool", "false",
               "Generate a more organized IR for better testing(warning: highly inefficient)">
    ];

    let constructor = "mlir::modelica::createModelLegalizationPass()";
}

def MatchingPass : Pass<"match", "mlir::ModuleOp"> {
    let summary = "Perform the matching process on a model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model onto which perform the matching">,
        Option<"processICModel", "process-ic-model", "bool", "true",
               "Process the 'initial conditions' model">,
        Option<"processMainModel", "process-main-model", "bool", "true",
               "Process the 'main' model">,
        Option<"debugView", "debug-view", "bool", "false",
               "Generate a more organized IR for better testing(warning: highly inefficient)">
    ];

    let constructor = "mlir::modelica::createMatchingPass()";
}

def CyclesSolvingPass : Pass<"solve-cycles", "mlir::ModuleOp"> {
    let summary = "Solve the cycles of matched model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model whose cycles hs to be solved">,
        Option<"solver", "solver", "marco::codegen::Solver", "\"forwardEuler\"",
               "Set the solver to be used for the simulation">,
        Option<"processICModel", "process-ic-model", "bool", "true",
               "Process the 'initial conditions' model">,
        Option<"processMainModel", "process-main-model", "bool", "true",
               "Process the 'main' model">,
        Option<"debugView", "debug-view", "bool", "false",
               "Generate a more organized IR for better testing(warning: highly inefficient)">
    ];

    let constructor = "mlir::modelica::createCyclesSolvingPass()";
}

def ConstantFoldingPass : Pass<"constant-folding", "mlir::ModuleOp"> {
  let summary = "Fold the constants of the model";

  let description = [{
      // TODO
  }];

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect"
  ];

  let options = [
    Option<"modelName", "model-name", "std::string", "",
           "Set the model whose cycles hs to be solved">,
    Option<"debugView", "debug-view", "bool", "false",
           "Generate a more organized IR for better testing (warning: highly inefficient)">
  ];

  let constructor = "mlir::modelica::createConstantFoldingPass()";
}

def SchedulingPass : Pass<"schedule", "mlir::ModuleOp"> {
    let summary = "Perform the scheduling process on a matched model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model onto which perform the scheduling">,
        Option<"processICModel", "process-ic-model", "bool", "true",
               "Process the 'initial conditions' model">,
        Option<"processMainModel", "process-main-model", "bool", "true",
               "Process the 'main' model">,
        Option<"debugView", "debug-view", "bool", "false",
               "Generate a more organized IR for better testing(warning: highly inefficient)">
    ];

    let constructor = "mlir::modelica::createSchedulingPass()";
}

def ModelConversionPass : Pass<"convert-model", "mlir::ModuleOp"> {
    let summary = "TODO";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::ida::IDADialect",
        "mlir::arith::ArithmeticDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">,
        Option<"model", "model", "std::string", "",
               "Set the model from which generate the simulation">,
        Option<"processICModel", "process-ic-model", "bool", "true",
               "Enable the processing of the 'initial conditions' model">,
        Option<"processMainModel", "process-main-model", "bool", "true",
               "Enable the processing of the 'main' model">,
        Option<"startTime", "start-time", "double", "0",
               "Set the simulation start time">,
        Option<"endTime", "end-time", "double", "1",
               "Set the simulation end time">,
        Option<"timeStep", "time-step", "double", "0.1",
               "Set the simulation time step">,
        Option<"emitSimulationMainFunction", "emit-simulation-main-function", "bool", "true",
               "Emit the 'main' function of the simulation">,
        Option<"variablesFilter", "simulation-variables-filter", "std::string", "",
               "Set a print filter for the variables of the simulation">,
        Option<"solver", "solver", "marco::codegen::Solver", "\"forwardEuler\"",
               "Set a the solver to be used for the simulation">,
        Option<"idaEquidistantTimeGrid", "ida-equidistant-time-grid", "bool", "false",
               "Set the IDA solver to use an equidistant time grid">
    ];

    let constructor = "mlir::modelica::createModelConversionPass()";
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
