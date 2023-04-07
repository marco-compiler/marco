#ifndef MARCO_CODEGEN_CONVERSION_PASSES_TD
#define MARCO_CODEGEN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// IDA to Func
//===---------------------------------------------------------------------===//

def IDAToFuncConversionPass : Pass<"convert-ida-to-func", "mlir::ModuleOp"> {
    let summary = "IDA: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">
    ];

    let constructor = "mlir::createIDAToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// IDA to LLVM
//===---------------------------------------------------------------------===//

def IDAToLLVMConversionPass : Pass<"convert-ida-to-llvm", "mlir::ModuleOp"> {
    let summary = "IDA: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createIDAToLLVMConversionPass()";
}

//===----------------------------------------------------------------------===//
// KINSOL to LLVM
//===----------------------------------------------------------------------===//

def KINSOLToLLVMConversionPass : Pass<"convert-kinsol-to-llvm", "mlir::ModuleOp"> {
    let summary = "KINSOL: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createKINSOLToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to Arith
//===---------------------------------------------------------------------===//

def ModelicaToArithConversionPass : Pass<"convert-modelica-to-arith", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Arith dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createModelicaToArithConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to CF
//===---------------------------------------------------------------------===//

def ModelicaToCFConversionPass : Pass<"convert-modelica-to-cf", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to CF dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::memref::MemRefDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"outputArraysPromotion", "output-arrays-promotion", "bool",
               "true", "Promote the output static arrays to arguments">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createModelicaToCFConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to Func
//===---------------------------------------------------------------------===//

def ModelicaToFuncConversionPass : Pass<"convert-modelica-to-func", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::ida::IDADialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::modelica::ModelicaDialect",
        "mlir::memref::MemRefDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">,
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">
    ];

    let constructor = "mlir::createModelicaToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to LLVM
//===---------------------------------------------------------------------===//

def ModelicaToLLVMConversionPass : Pass<"convert-modelica-to-llvm", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::ida::IDADialect",
        "mlir::kinsol::KINSOLDialect",
        "mlir::simulation::SimulationDialect",
        "mlir::arith::ArithDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createModelicaToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to MemRef
//===---------------------------------------------------------------------===//

def ModelicaToMemRefConversionPass : Pass<"convert-modelica-to-memref", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to MemRef dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::memref::MemRefDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createModelicaToMemRefConversionPass()";
}

//===---------------------------------------------------------------------===//
// Modelica to Vector
//===---------------------------------------------------------------------===//

def ModelicaToVectorConversionPass : Pass<"convert-modelica-to-vector", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Vector dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::vector::VectorDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
    ];

    let constructor = "mlir::createModelicaToVectorConversionPass()";
}

//===---------------------------------------------------------------------===//
// Simulation to Func
//===---------------------------------------------------------------------===//

def SimulationToFuncConversionPass : Pass<"convert-simulation-to-func", "mlir::ModuleOp"> {
    let summary = "Simulation: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"emitMainFunction", "emit-main-function", "bool", "true",
               "Emit the 'main' function of the simulation">
    ];

    let constructor = "mlir::createSimulationToFuncConversionPass()";
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
