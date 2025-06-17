#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MATCHING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MATCHING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_MATCHINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

/// Create a pass performing the matching process on a model.
std::unique_ptr<mlir::Pass> createMatchingPass();

/// Create a pass performing the matching process on a model.
std::unique_ptr<mlir::Pass>
createMatchingPass(const MatchingPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MATCHING_H
