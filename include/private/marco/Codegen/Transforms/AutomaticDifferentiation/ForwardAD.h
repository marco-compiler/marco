#ifndef MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
#define MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"
#include <set>

namespace marco::codegen
{
  class ForwardAD
  {
    public:
      /// Check if an operation has already been already derived.
      bool isDerived(mlir::Operation* op) const;

      /// Set an operation as already derived.
      void setAsDerived(mlir::Operation* op);

      mlir::LogicalResult createFullDerFunction(
          mlir::OpBuilder& builder, mlir::modelica::FunctionOp functionOp);

      mlir::LogicalResult createPartialDerFunction(
          mlir::OpBuilder& builder, mlir::modelica::DerFunctionOp derFunctionOp);

      mlir::ValueRange deriveTree(
          mlir::OpBuilder& builder, mlir::modelica::DerivableOpInterface op, mlir::BlockAndValueMapping& derivatives);

      mlir::modelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder, mlir::modelica::DerFunctionOp derFunctionOp);

      mlir::modelica::FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::modelica::FunctionOp functionOp,
          llvm::StringRef derivedFunctionName);

      std::pair<mlir::modelica::FunctionOp, unsigned int> getPartialDerBaseFunction(
          mlir::modelica::FunctionOp functionOp);

      mlir::LogicalResult deriveFunctionBody(
          mlir::OpBuilder& builder,
          mlir::modelica::FunctionOp functionOp,
          mlir::BlockAndValueMapping& derivatives,
          std::function<mlir::ValueRange(mlir::OpBuilder&, mlir::modelica::CallOp, mlir::BlockAndValueMapping&)> deriveFn);

      mlir::ValueRange createCallOpFullDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::CallOp callOp,
          mlir::BlockAndValueMapping& derivatives);

      mlir::ValueRange createCallOpPartialDerivative(
          mlir::OpBuilder& builder,
          mlir::modelica::CallOp callOp,
          mlir::BlockAndValueMapping& derivatives);

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
