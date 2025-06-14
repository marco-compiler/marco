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

def IDAToFuncConversionPass : Pass<"convert-ida-to-func", "mlir::ModuleOp">
{
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

    let constructor = "mlir::createIDAToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// IDA to LLVM
//===---------------------------------------------------------------------===//

def IDAToLLVMConversionPass : Pass<"convert-ida-to-llvm", "mlir::ModuleOp">
{
    let summary = "IDA: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
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

    let constructor = "mlir::createKINSOLToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Arith
//===---------------------------------------------------------------------===//

def BaseModelicaToArithConversionPass
    : Pass<"convert-bmodelica-to-arith", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to Arith dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::createBaseModelicaToArithConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to CF
//===---------------------------------------------------------------------===//

def BaseModelicaToCFConversionPass
    : Pass<"convert-bmodelica-to-cf", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to CF dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::cf::ControlFlowDialect"
    ];

    let constructor = "mlir::createBaseModelicaToCFConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Func
//===---------------------------------------------------------------------===//

def BaseModelicaToFuncConversionPass
    : Pass<"convert-bmodelica-to-func", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::createBaseModelicaToFuncConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica Raw Variables conversion
//===---------------------------------------------------------------------===//

def BaseModelicaRawVariablesConversionPass
    : Pass<"convert-bmodelica-raw-variables">
{
    let summary = "Base Modelica: conversion to Func dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::memref::MemRefDialect"
    ];

    let constructor = "mlir::createBaseModelicaRawVariablesConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Linalg
//===---------------------------------------------------------------------===//

def BaseModelicaToLinalgConversionPass
    : Pass<"convert-bmodelica-to-linalg", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to Linalg dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect"
    ];

    let constructor = "mlir::createBaseModelicaToLinalgConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to LLVM
//===---------------------------------------------------------------------===//

def BaseModelicaToLLVMConversionPass
    : Pass<"convert-bmodelica-to-llvm", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createBaseModelicaToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to MemRef
//===---------------------------------------------------------------------===//

def BaseModelicaToMemRefConversionPass
    : Pass<"convert-bmodelica-to-memref", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to MemRef dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::scf::SCFDialect",
        "mlir::memref::MemRefDialect"
    ];

    let constructor = "mlir::createBaseModelicaToMemRefConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Tensor
//===---------------------------------------------------------------------===//

def BaseModelicaToTensorConversionPass
    : Pass<"convert-bmodelica-to-tensor", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to Tensor dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect"
    ];

    let constructor = "mlir::createBaseModelicaToTensorConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Runtime
//===---------------------------------------------------------------------===//

def BaseModelicaToRuntimeConversionPass
    : Pass<"convert-bmodelica-to-runtime", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to Runtime dialect";

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::runtime::RuntimeDialect",
        "mlir::func::FuncDialect"
    ];

    let options = [
        Option<"variableFilter", "variable-filter",
               "std::string", "",
               "Set a print filter for the variables of the simulation">,
    ];

    let constructor = "mlir::createBaseModelicaToRuntimeConversionPass()";
}

//===---------------------------------------------------------------------===//
// Runtime to Func
//===---------------------------------------------------------------------===//

def RuntimeToFuncConversionPass
    : Pass<"convert-runtime-to-func", "mlir::ModuleOp">
{
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
// Model runtime metadata conversion
//===---------------------------------------------------------------------===//

def RuntimeModelMetadataConversionPass
    : Pass<"convert-runtime-model-metadata", "mlir::ModuleOp">
{
    let summary = "TODO";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createRuntimeModelMetadataConversionPass()";
}

//===---------------------------------------------------------------------===//
// Runtime to LLVM
//===---------------------------------------------------------------------===//

def RuntimeToLLVMConversionPass
    : Pass<"convert-runtime-to-llvm", "mlir::ModuleOp">
{
    let summary = "Runtime: conversion to LLVM dialect";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createRuntimeToLLVMConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to MLIR core dialects
//===---------------------------------------------------------------------===//

def BaseModelicaToMLIRCoreConversionPass
    : Pass<"convert-bmodelica-to-mlir-core", "mlir::ModuleOp">
{
    let summary = "Convert Base Modelica to MLIR core dialects";

    let description = [{
        Convert Base Modelica to MLIR core dialects.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::func::FuncDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::memref::MemRefDialect",
        "mlir::scf::SCFDialect",
        "mlir::runtime::RuntimeDialect",
        "mlir::tensor::TensorDialect"
    ];

    let constructor = "mlir::createBaseModelicaToMLIRCoreConversionPass()";
}

//===---------------------------------------------------------------------===//
// Base Modelica to Runtime Call
//===---------------------------------------------------------------------===//

def BaseModelicaToRuntimeCallConversionPass
    : Pass<"convert-bmodelica-to-runtime-call", "mlir::ModuleOp">
{
    let summary = "Base Modelica: conversion to runtime calls";

    let description = [{
        Convert the Base Modelica operations to calls to functions living in
        the runtime library.
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::memref::MemRefDialect",
        "mlir::runtime::RuntimeDialect",
        "mlir::tensor::TensorDialect"
    ];

    let constructor = "mlir::createBaseModelicaToRuntimeCallConversionPass()";
}

def AllToLLVMConversionPass : Pass<"convert-all-to-llvm"> {
    let summary = "Convert to LLVM dialect";

    let description = [{
        Convert the operations to the LLVM dialect.
        The pass is meant to be used in the final stage of the conversion pipeline,
        right before folding unrealized casts and legalizing the IR for export.
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];

    let constructor = "mlir::createAllToLLVMConversionPass()";
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_TD
