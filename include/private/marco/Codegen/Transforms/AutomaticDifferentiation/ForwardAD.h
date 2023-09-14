#ifndef MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
#define MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"
#include <set>

namespace mlir::modelica
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
          mlir::modelica::FunctionOp functionOp,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult convertPartialDerFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::DerFunctionOp derFunctionOp,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::ValueRange deriveTree(
          mlir::OpBuilder& builder,
          mlir::modelica::DerivableOpInterface op,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& derivatives);

      mlir::modelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::modelica::DerFunctionOp derFunctionOp);

      mlir::modelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::SymbolTableCollection& symbolTable,
          mlir::modelica::FunctionOp functionOp,
          llvm::StringRef derivedFunctionName);

      std::pair<mlir::modelica::FunctionOp, unsigned int> getPartialDerBaseFunction(
          mlir::modelica::FunctionOp functionOp);

      mlir::LogicalResult deriveRegion(
          mlir::OpBuilder& builder,
          mlir::Region& region,
          mlir::SymbolTableCollection& symbolTable,
          llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives,
          llvm::function_ref<mlir::ValueRange(
              mlir::OpBuilder&,
              mlir::Operation*,
              mlir::SymbolTableCollection&,
              const llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>&,
              mlir::IRMapping&)> deriveFn);

      bool isDerivable(mlir::Operation* op) const;

      mlir::ValueRange createOpFullDerivative(
          mlir::OpBuilder& builder,
          mlir::Operation* op,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives);

      mlir::ValueRange createOpPartialDerivative(
          mlir::OpBuilder& builder,
          mlir::Operation* op,
          mlir::SymbolTableCollection& symbolTable,
          const llvm::DenseMap<
              mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
          mlir::IRMapping& ssaDerivatives);

      mlir::ValueRange createCallOpFullDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::CallOp callOp,
          mlir::IRMapping& ssaDerivatives);

      mlir::ValueRange createCallOpPartialDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::CallOp callOp,
          mlir::SymbolTableCollection& symbolTable,
          mlir::IRMapping& ssaDerivatives);

      mlir::ValueRange createTimeOpFullDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::TimeOp timeOp,
          mlir::IRMapping& ssaDerivatives);

      mlir::ValueRange createTimeOpPartialDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::TimeOp timeOp,
          mlir::IRMapping& ssaDerivatives);

    private:
      // Keeps track of the operations that have already been derived
      std::set<mlir::Operation*> derivedOps;

      // Map each partial derivative template function to its base function
      llvm::StringMap<mlir::modelica::FunctionOp> partialDerTemplates;

      llvm::StringMap<mlir::modelica::FunctionOp> partialDersTemplateCallers;

      llvm::StringMap<mlir::ArrayAttr> partialDerTemplatesIndependentVars;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
