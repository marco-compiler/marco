#include "marco/Codegen/Transforms/SCCAbsenceVerification.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SCCABSENCEVERIFICATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class SCCAbsenceVerification
      : public impl::SCCAbsenceVerificationPassBase<SCCAbsenceVerification>
  {
    public:
      using SCCAbsenceVerificationPassBase<SCCAbsenceVerification>
          ::SCCAbsenceVerificationPassBase;

      void runOnOperation() override;
  };
}

void SCCAbsenceVerification::runOnOperation()
{
  llvm::SmallVector<SCCOp> SCCs;

  getOperation().walk([&](SCCOp scc) {
    SCCs.push_back(scc);
  });

  if (!SCCs.empty()) {
    for (SCCOp scc : SCCs) {
      scc.emitError() << "unsolved SCC";
    }

    signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSCCAbsenceVerificationPass()
  {
    return std::make_unique<SCCAbsenceVerification>();
  }
}