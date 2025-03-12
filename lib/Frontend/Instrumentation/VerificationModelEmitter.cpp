#include "marco/Frontend/Instrumentation/VerificationModelEmitter.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::frontend;

namespace marco::frontend {
VerificationModelEmitter::VerificationModelEmitter(
    mlir::Pass *afterPass, std::unique_ptr<llvm::raw_ostream> os)
    : afterPass(afterPass), os(std::move(os)) {}

VerificationModelEmitter::~VerificationModelEmitter() = default;

void VerificationModelEmitter::runAfterPass(mlir::Pass *pass,
                                            mlir::Operation *op) {
  if (pass == afterPass) {
    assert(executionCounts == 0 && "Verification model already emitted");

    if (os) {
      op->print(*os);
      os->flush();
    } else {
      llvm::errs() << "Can't emit MLIR model for verification\n";
    }

    ++executionCounts;
  }
}
} // namespace marco::frontend
