#ifndef MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
#define MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createNegateOpDistributionPass();

  inline void registerNegateOpDistributionPass()
  {
    mlir::registerPass(
        "distribute-neg", "Distribute the negation operations",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createNegateOpDistributionPass();
        });
  }

  std::unique_ptr<mlir::Pass> createMulOpDistributionPass();

  inline void registerMulOpDistributionPass()
  {
    mlir::registerPass(
        "distribute-mul", "Distribute the multiplication operations",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createMulOpDistributionPass();
        });
  }

  std::unique_ptr<mlir::Pass> createDivOpDistributionPass();

  inline void registerDivOpDistributionPass()
  {
    mlir::registerPass(
        "distribute-div", "Distribute the division operations",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createDivOpDistributionPass();
        });
  }
}

#endif // MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
