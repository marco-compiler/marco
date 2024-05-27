#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::bmodelica::ad::forward
{
  //std::pair<FunctionOp, uint64_t>
  //getPartialDerBaseFunction(const State& state, FunctionOp functionOp);

  std::string getPartialDerFunctionName(llvm::StringRef baseName);

  std::string getPartialDerVariableName(
      llvm::StringRef baseName, uint64_t order);

  std::optional<FunctionOp> createFunctionPartialDerivative(
      mlir::OpBuilder& builder,
      State& state,
      FunctionOp functionOp,
      llvm::StringRef derivativeName);

  std::string getTimeDerFunctionName(llvm::StringRef baseName);

  /// Compose the full derivative member name according to the derivative order.
  /// If the order is 1, then it is omitted.
  ///
  /// @param variableName  base variable name
  /// @param order 				 derivative order
  /// @return derived variable name
  std::string getTimeDerVariableName(
      llvm::StringRef baseName, uint64_t order);

  /// Given a full derivative variable name of order n, compose the name of the
  /// n + 1 variable order.
  ///
  /// @param currentName 	   current variable name
  /// @param requestedOrder  desired derivative order
  /// @return next order derived variable name
  std::string getNextTimeDerVariableName(
      llvm::StringRef currentName, uint64_t requestedOrder);

  bool isTimeDerivative(
      llvm::StringRef name, FunctionOp functionOp, uint64_t maxOrder);

  void mapTimeDerivativeFunctionVariables(
      FunctionOp functionOp,
      ad::forward::State& state);

  std::optional<FunctionOp> createFunctionTimeDerivative(
      mlir::OpBuilder& builder,
      State& state,
      FunctionOp functionOp,
      uint64_t functionOrder,
      llvm::StringRef derivativeName,
      uint64_t derivativeOrder);
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_FORWARD_AD_H
