#ifndef MARCO_CODEGEN_CONVERSION_PASSES_TD
#define MARCO_CODEGEN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IDA
//===----------------------------------------------------------------------===//

def IDAToLLVM : Pass<"ida-to-llvm", "mlir::ModuleOp"> {
  let summary = "IDA: conversion to Std + LLVM dialect";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createIDAToLLVMConversionPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Modelica
//===----------------------------------------------------------------------===//

def ConvertModelica : Pass<"convert-modelica-to-llvm", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to std + scf + llvm dialects";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createModelicaConversionPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertModelicaToCFG : Pass<"convert-modelica-to-cfg", "mlir::ModuleOp"> {
  let summary = "Modelica: convert to CFG";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createLowerToCFGPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertModelicaToLLVM : Pass<"convert-to-llvm", "mlir::ModuleOp"> {
  let summary = "Modelica: LLVM lowering pass";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createLLVMLoweringPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
