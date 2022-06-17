#ifndef MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
#define MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createNegateOpDistributionPass();

  std::unique_ptr<mlir::Pass> createMulOpDistributionPass();

  std::unique_ptr<mlir::Pass> createDivOpDistributionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
