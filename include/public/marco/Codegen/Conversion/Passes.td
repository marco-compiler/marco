#ifndef MARCO_CODEGEN_CONVERSION_PASSES_TD
#define MARCO_CODEGEN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// SUNDIALS to Func
//===---------------------------------------------------------------------===//

def SUNDIALSToFuncConversionPass
    : Pass<"convert-sundials-to-func", "mlir::ModuleOp">
{
    let summary = "SUNDIALS: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createSUNDIALSToFuncConversionPass()";
}

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

//===---------------------------------------------------------------------===//
// KINSOL to Func
//===---------------------------------------------------------------------===//

def KINSOLToFuncConversionPass
    : Pass<"convert-kinsol-to-func", "mlir::ModuleOp">
{
    let summary = "KINSOL: conversion to Func dialect";

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

    let constructor = "mlir::createKINSOLToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// KINSOL to LLVM
//===---------------------------------------------------------------------===//

def KINSOLToLLVMConversionPass
    : Pass<"convert-kinsol-to-llvm", "mlir::ModuleOp">
{
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
// Base Modelica to Arith
//===---------------------------------------------------------------------===//

def BaseModelicaToArithConversionPass : Pass<"convert-bmodelica-to-arith", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Arith dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
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

    let constructor = "mlir::createBaseModelicaToArithConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to CF
//===---------------------------------------------------------------------===//

def BaseModelicaToCFConversionPass : Pass<"convert-bmodelica-to-cf", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to CF dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
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

    let constructor = "mlir::createBaseModelicaToCFConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Func
//===---------------------------------------------------------------------===//

def BaseModelicaToFuncConversionPass : Pass<"convert-bmodelica-to-func", "mlir::ModuleOp"> {
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
        "mlir::bmodelica::BaseModelicaDialect",
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

    let constructor = "mlir::createBaseModelicaToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to LLVM
//===---------------------------------------------------------------------===//

def BaseModelicaToLLVMConversionPass : Pass<"convert-bmodelica-to-llvm", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::ida::IDADialect",
        "mlir::runtime::RuntimeDialect",
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

    let constructor = "mlir::createBaseModelicaToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to MemRef
//===---------------------------------------------------------------------===//

def BaseModelicaToMemRefConversionPass : Pass<"convert-bmodelica-to-memref", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to MemRef dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
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

    let constructor = "mlir::createBaseModelicaToMemRefConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Vector
//===---------------------------------------------------------------------===//

def BaseModelicaToVectorConversionPass : Pass<"convert-bmodelica-to-vector", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Vector dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::vector::VectorDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
    ];

    let constructor = "mlir::createBaseModelicaToVectorConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Runtime
//===---------------------------------------------------------------------===//

def BaseModelicaToRuntimeConversionPass : Pass<"convert-bmodelica-to-runtime", "mlir::ModuleOp"> {
    let summary = "Modelica: conversion to Runtime dialect";

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::runtime::RuntimeDialect",
        "mlir::func::FuncDialect"
    ];

    let options = [
        Option<"variablesFilter", "variables-filter",
               "std::string", "",
               "Set a print filter for the variables of the simulation">,
    ];

    let constructor = "mlir::createBaseModelicaToRuntimeConversionPass()";
}

//===---------------------------------------------------------------------===//
// Runtime to Func
//===---------------------------------------------------------------------===//

def RuntimeToFuncConversionPass : Pass<"convert-runtime-to-func", "mlir::ModuleOp"> {
    let summary = "Runtime: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createRuntimeToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// Runtime to LLVM
//===---------------------------------------------------------------------===//

def RuntimeToLLVMConversionPass
    : Pass<"convert-runtime-to-llvm", "mlir::ModuleOp"> {
    let summary = "Runtime: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let options = [
        Option<"bitWidth", "bitWidth", "unsigned int", "64",
               "Specify the bit-width for the simulation data">,
        Option<"dataLayout", "dataLayout", "std::string", "",
               "Specify the data layout">
    ];

    let constructor = "mlir::createRuntimeToLLVMConversionPass()";
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
