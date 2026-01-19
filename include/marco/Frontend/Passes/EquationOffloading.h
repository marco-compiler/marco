#ifndef MARCO_FRONTEND_PASSES_EQUATIONOFFLOADING_H
#define MARCO_FRONTEND_PASSES_EQUATIONOFFLOADING_H

#include "marco/Frontend/Passes/EquationTargets/EquationTarget.h"
#include "mlir/Pass/Pass.h"

namespace marco::frontend {
struct EquationOffloadingPassOptions {
  llvm::SmallVector<std::unique_ptr<EquationTarget>> targets;
  bool attachCandidateTargetsAsAttribute{true};

  EquationOffloadingPassOptions();
  EquationOffloadingPassOptions(const EquationOffloadingPassOptions &other);
  EquationOffloadingPassOptions(EquationOffloadingPassOptions &&other) noexcept;

  EquationOffloadingPassOptions &
  operator=(const EquationOffloadingPassOptions &other);

  EquationOffloadingPassOptions &
  operator=(EquationOffloadingPassOptions &&other) noexcept;
};

std::unique_ptr<mlir::Pass>
createEquationOffloadingPass(EquationOffloadingPassOptions options);
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_EQUATIONOFFLOADING_H
