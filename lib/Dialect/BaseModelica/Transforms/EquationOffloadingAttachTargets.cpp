#include "marco/Dialect/BaseModelica/Transforms/EquationOffloadingAttachTargets.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONOFFLOADINGATTACHTARGETSPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationOffloadingAttachTargetsPass
    : public impl::EquationOffloadingAttachTargetsPassBase<
          EquationOffloadingAttachTargetsPass> {
public:
  using EquationOffloadingAttachTargetsPassBase::
      EquationOffloadingAttachTargetsPassBase;

  void runOnOperation() override;
};
} // namespace

void EquationOffloadingAttachTargetsPass::runOnOperation() {
  llvm::StringSet<> uniqueTargets;

  for (const auto &target : targets) {
    uniqueTargets.insert(target);
  }

  llvm::SmallVector<llvm::StringRef> sortedTargets;

  for (const auto &target : uniqueTargets) {
    sortedTargets.push_back(target.getKey());
  }

  llvm::sort(sortedTargets);

  getOperation().walk([&](EquationCallOp callOp) {
    llvm::SmallVector<mlir::Attribute> targets;
    targets.push_back(mlir::StringAttr::get(&getContext(), "cpu"));

    if (auto offloadInt =
            mlir::dyn_cast<OffloadInterface>(callOp.getOperation())) {
      for (llvm::StringRef target : sortedTargets) {
        if (offloadInt.isOffloadable(target)) {
          targets.push_back(mlir::StringAttr::get(&getContext(), target));
        }
      }
    }

    callOp.setTargetsAttr(mlir::ArrayAttr::get(&getContext(), targets));
  });

  // Temporarily restrict the possible targets to just one choice, assuming that
  // the only alternative to CPU execution is GPU offloading.
  // TODO In the long term, we should keep all the attached targets, and decide
  // and dynamically decide the offloading target at runtime.
  // at runtime which one ot use for offloading. This would require emitting the
  // code for all the targets, but would enable online tuning.
  // This should be done as part of a bigger effort aiming to communicate to
  // the whole dependency graph to the runtime environment.
  // When this design will be implemented, then the following code will be
  // safely removable.

  getOperation().walk([](EquationCallOp callOp) {
    bool hasGPU = false;

    for (mlir::Attribute target : callOp.getTargets()) {
      if (mlir::cast<mlir::StringAttr>(target) == "gpu") {
        hasGPU = true;
        break;
      }
    }

    llvm::SmallVector<mlir::Attribute, 1> targets;

    targets.push_back(
        mlir::StringAttr::get(callOp.getContext(), hasGPU ? "gpu" : "cpu"));

    callOp.setTargetsAttr(mlir::ArrayAttr::get(callOp.getContext(), targets));
  });
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationOffloadingAttachTargetsPass() {
  return std::make_unique<EquationOffloadingAttachTargetsPass>();
}

std::unique_ptr<mlir::Pass> createEquationOffloadingAttachTargetsPass(
    const EquationOffloadingAttachTargetsPassOptions &options) {
  return std::make_unique<EquationOffloadingAttachTargetsPass>(options);
}
} // namespace mlir::bmodelica
