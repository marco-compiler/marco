#include "marco/Dialect/BaseModelica/Transforms/SCCAbsenceVerification.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCCABSENCEVERIFICATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class SCCAbsenceVerification
    : public impl::SCCAbsenceVerificationPassBase<SCCAbsenceVerification> {
public:
  using SCCAbsenceVerificationPassBase<
      SCCAbsenceVerification>::SCCAbsenceVerificationPassBase;

  void runOnOperation() override;
};
} // namespace

void SCCAbsenceVerification::runOnOperation() {
  llvm::SmallVector<SCCOp> SCCs;

  getOperation().walk([&](SCCOp scc) { SCCs.push_back(scc); });

  if (!SCCs.empty()) {
    for (SCCOp scc : SCCs) {
      scc.emitError() << "unsolved SCC";
    }

    signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSCCAbsenceVerificationPass() {
  return std::make_unique<SCCAbsenceVerification>();
}
} // namespace mlir::bmodelica
