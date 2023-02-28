#ifndef MARCO_CODEGEN_VERIFIER_H
#define MARCO_CODEGEN_VERIFIER_H

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace marco::codegen::lowering
{
  /// A verification pass to verify the output from the bridge. This provides a
  /// little bit of glue to run a verifier pass directly.
  class VerifierPass
      : public mlir::PassWrapper<VerifierPass, mlir::OperationPass<>>
  {
    void runOnOperation() override;
  };

}

#endif // MARCO_CODEGEN_VERIFIER_H
