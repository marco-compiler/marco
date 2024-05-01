#ifndef MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
#define MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H

#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"
#include <set>

namespace mlir::bmodelica
{
  class ForwardAD
  {
    public:
      /// Check if an operation has already been already derived.
      bool isDerived(mlir::Operation* op) const;

      /// Set an operation as already derived.
      void setAsDerived(mlir::Operation* op);

      mlir::LogicalResult createFullDerFunction(
          mlir::OpBuilder& builder,
          mlir::bmodelica::FunctionOp functionOp,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult convertPartialDerFunction(
          mlir::OpBuilder& builder,
          mlir::bmodelica::DerFunctionOp derFunctionOp,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult deriveTree(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::bmodelica::DerivableOpInterface op,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& derivatives);

      mlir::bmodelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::bmodelica::DerFunctionOp derFunctionOp);

      mlir::bmodelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::SymbolTableCollection& symbolTable,
          mlir::bmodelica::FunctionOp functionOp,
          llvm::StringRef derivedFunctionName);

      std::pair<mlir::bmodelica::FunctionOp, unsigned int> getPartialDerBaseFunction(
          mlir::bmodelica::FunctionOp functionOp);

      mlir::LogicalResult deriveRegion(
          mlir::OpBuilder& builder,
          mlir::Region& region,
          mlir::SymbolTableCollection& symbolTable,
          llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives,
          llvm::function_ref<mlir::LogicalResult(
              llvm::SmallVectorImpl<mlir::Value>& results,
              mlir::OpBuilder&,
              mlir::Operation*,
              mlir::SymbolTableCollection&,
              const llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>&,
              mlir::IRMapping&)> deriveFn);

      bool isDerivable(mlir::Operation* op) const;

      mlir::LogicalResult createOpFullDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::Operation* op,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives);

      mlir::LogicalResult createOpPartialDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::Operation* op,
          mlir::SymbolTableCollection& symbolTable,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives);

      mlir::LogicalResult createCallOpFullDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::bmodelica::CallOp callOp,
          mlir::IRMapping& ssaDerivatives);

      mlir::LogicalResult createCallOpPartialDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::bmodelica::CallOp callOp,
          mlir::SymbolTableCollection& symbolTable,
          mlir::IRMapping& ssaDerivatives);

      mlir::LogicalResult createTimeOpFullDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::bmodelica::TimeOp timeOp);

      mlir::LogicalResult createTimeOpPartialDerivative(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder& builder,
          mlir::bmodelica::TimeOp timeOp);

    private:
      // Keeps track of the operations that have already been derived
      std::set<mlir::Operation*> derivedOps;

      // Map each partial derivative template function to its base function
      llvm::StringMap<mlir::bmodelica::FunctionOp> partialDerTemplates;

      llvm::StringMap<mlir::bmodelica::FunctionOp> partialDersTemplateCallers;

      llvm::StringMap<mlir::ArrayAttr> partialDerTemplatesIndependentVars;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
