#ifndef MARCO_CODEGEN_CONVERSION_PASSES_TD
#define MARCO_CODEGEN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IDA
//===----------------------------------------------------------------------===//

def IDAToLLVM : Pass<"convert-ida-to-llvm", "mlir::ModuleOp"> {
  let summary = "IDA: conversion to Std + LLVM dialect";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createIDAToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Modelica to LLVM
//===----------------------------------------------------------------------===//

def ModelicaToLLVM : Pass<"convert-modelica-to-llvm", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to std + scf + llvm dialects";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createModelicaToLLVMPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ModelicaToCF : Pass<"convert-modelica-to-cf", "mlir::ModuleOp"> {
  let summary = "Modelica: convert to CFG";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createModelicaToCFPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
