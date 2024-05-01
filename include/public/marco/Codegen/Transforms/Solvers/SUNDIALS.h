#ifndef MARCO_CODEGEN_TRANSFORMS_SOLVERS_SUNDIALS_H
#define MARCO_CODEGEN_TRANSFORMS_SOLVERS_SUNDIALS_H

#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "marco/Dialect/SUNDIALS/SUNDIALSDialect.h"

namespace mlir::bmodelica
{
  mlir::sundials::VariableGetterOp createGetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      VariableOp variable,
      llvm::StringRef functionName);

  mlir::sundials::VariableSetterOp createSetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      VariableOp variable,
      llvm::StringRef functionName);

  GlobalVariableOp createGlobalADSeed(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::StringRef name,
      mlir::Type type);

  void setGlobalADSeed(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      GlobalVariableOp seedVariableOp,
      mlir::ValueRange indices,
      mlir::Value value);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SOLVERS_SUNDIALS_H
