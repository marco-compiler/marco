#include "marco/Codegen/Verifier.h"

namespace marco::codegen::lowering
{
  void VerifierPass::runOnOperation()
  {
    if (mlir::failed(mlir::verify(getOperation()))) {
      signalPassFailure();
    }

    markAllAnalysesPreserved();
  }
}
