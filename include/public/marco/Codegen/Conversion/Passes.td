#ifndef MARCO_CODEGEN_CONVERSION_PASSES_TD
#define MARCO_CODEGEN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IDA to LLVM
//===----------------------------------------------------------------------===//

def IDAToLLVM : Pass<"convert-ida-to-llvm", "mlir::ModuleOp"> {
  let summary = "IDA: conversion to LLVM dialect";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createIDAToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Modelica to Arith
//===----------------------------------------------------------------------===//

def ModelicaToArith : Pass<"convert-modelica-to-arith", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to Arith dialect";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createModelicaToArithPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Modelica to CF
//===----------------------------------------------------------------------===//

def ModelicaToCF : Pass<"convert-modelica-to-cf", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to CF dialect";

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

//===----------------------------------------------------------------------===//
// Modelica to LLVM
//===----------------------------------------------------------------------===//

def ModelicaToLLVM : Pass<"convert-modelica-to-llvm", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to LLVM dialect";

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
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Modelica to MemRef
//===----------------------------------------------------------------------===//

def ModelicaToMemRef : Pass<"convert-modelica-to-memref", "mlir::ModuleOp"> {
  let summary = "Modelica: conversion to MemRef dialect";

  let description = [{
    // TODO
  }];

  let constructor = "marco::codegen::createModelicaToMemRefPass()";

  let dependentDialects = [
    "mlir::modelica::ModelicaDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
