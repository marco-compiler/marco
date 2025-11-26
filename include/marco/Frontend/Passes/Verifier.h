#ifndef MARCO_FRONTEND_PASSES_VERIFIER_H
#define MARCO_FRONTEND_PASSES_VERIFIER_H

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace marco::frontend {
/// A verification pass to verify the output from the bridge. This provides a
/// little bit of glue to run a verifier pass directly.
class VerifierPass
    : public mlir::PassWrapper<VerifierPass, mlir::OperationPass<>> {
  void runOnOperation() override;
};

} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_VERIFIER_H
