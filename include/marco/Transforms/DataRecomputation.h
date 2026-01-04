#ifndef MARCO_TRANSFORMS_DATARECOMPUTATION_H
#define MARCO_TRANSFORMS_DATARECOMPUTATION_H


#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DATARECOMPUTATIONPASS
#include "marco/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDataRecomputationPass();

} // namespace mlir

#endif /* end of include guard: MARCO_TRANSFORMS_DATARECOMPUTATION_H */

