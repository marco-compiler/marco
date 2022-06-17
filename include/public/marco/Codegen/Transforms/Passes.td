#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_TD
#define MARCO_CODEGEN_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ArrayDeallocation : Pass<"array-deallocation", "mlir::ModuleOp"> {
  let summary = "Automatic array deallocation";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createArrayDeallocationPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
  ];
}

def AutomaticDifferentiation : Pass<"auto-diff", "mlir::ModuleOp"> {
  let summary = "Automatic differentiation of functions";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createAutomaticDifferentiationPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect"
  ];
}

def ExplicitCastInsertion : Pass<"explicit-cast-insertion", "mlir::ModuleOp"> {
  let summary = "Explicit cast insertion";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createExplicitCastInsertionPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::scf::SCFDialect"
  ];
}

def FunctionScalarization : Pass<"scalarize", "mlir::ModuleOp"> {
  let summary = "Scalarize";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createFunctionScalarizationPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ModelSolving : Pass<"solve-model", "mlir::ModuleOp"> {
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

  let constructor = "marco::codegen::createAutomaticDifferentiationPass()";
}

def Matching : Pass<"matching", "mlir::ModuleOp"> {
  let summary = "Perform the matching on the model";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createMatchingPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect"
  ];
}

def Scheduling : Pass<"scheduling", "mlir::ModuleOp"> {
  let summary = "Perform the scheduling on the model";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createSchedulingPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect"
  ];
}

def NegateOpDistributionPass : Pass<"distribute-neg", "mlir::ModuleOp"> {
  let summary = "Distribute the negation operations";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createNegateOpDistributionPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect"
  ];
}

def MulOpDistributionPass : Pass<"distribute-mul", "mlir::ModuleOp"> {
  let summary = "Distribute the multiplication operations";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createMulOpDistributionPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect"
  ];
}

def DivOpDistributionPass : Pass<"distribute-div", "mlir::ModuleOp"> {
  let summary = "Distribute the division operations";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createDivOpDistributionPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect"
  ];
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_TD
