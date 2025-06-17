#ifndef MARCO_FRONTEND_INSTRUMENTATION_VERIFICATIONMODELEMITTER_H
#define MARCO_FRONTEND_INSTRUMENTATION_VERIFICATIONMODELEMITTER_H

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringSet.h"

namespace llvm {
class raw_ostream;
}

namespace marco::frontend {
struct VerificationModelEmitter : public mlir::PassInstrumentation {
  VerificationModelEmitter(mlir::Pass *afterPass,
                           std::unique_ptr<llvm::raw_ostream> os);

  ~VerificationModelEmitter() override;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  mlir::Pass *afterPass;
  std::unique_ptr<llvm::raw_ostream> os;
  uint64_t executionCounts{0};
};

} // namespace marco::frontend

#endif // MARCO_FRONTEND_INSTRUMENTATION_VERIFICATIONMODELEMITTER_H
