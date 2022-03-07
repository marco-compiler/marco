#ifndef MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
#define MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  /// Compose the full derivative member name according to the derivative order.
  /// If the order is 1, then it is omitted.
  ///
  /// @param variableName  base variable name
  /// @param order 				 derivative order
  /// @return derived variable name
  std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order);

  /// Given a full derivative variable name of order n, compose the name of the
  /// n + 1 variable order.
  ///
  /// @param currentName 	   current variable name
  /// @param requestedOrder  desired derivative order
  /// @return next order derived variable name
  std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder);

	std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();

	inline void registerAutomaticDifferentiationPass()
	{
		mlir::registerPass(
        "auto-diff", "Modelica: automatic differentiation of functions",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createAutomaticDifferentiationPass();
        });
	}
}

#endif // MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
