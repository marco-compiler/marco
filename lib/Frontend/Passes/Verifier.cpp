#include "marco/Frontend/Passes/Verifier.h"

namespace marco::frontend {
void VerifierPass::runOnOperation() {
  if (mlir::failed(mlir::verify(getOperation()))) {
    signalPassFailure();
  }

  markAllAnalysesPreserved();
}
} // namespace marco::frontend
