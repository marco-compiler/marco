#ifndef MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H
#define MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createArrayDeallocationPass();
}

#endif // MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H
