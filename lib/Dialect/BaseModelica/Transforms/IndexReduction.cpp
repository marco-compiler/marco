#define DEBUG_TYPE "index-reduction"

#include "marco/Dialect/BaseModelica/Transforms/IndexReduction.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "llvm/Support/Debug.h"

#include <mlir/Support/LogicalResult.h>
namespace mlir::bmodelica {
#define GEN_PASS_DEF_INDEXREDUCTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class IndexReductionPass final
    : public impl::IndexReductionPassBase<IndexReductionPass> {
public:
  using IndexReductionPassBase::IndexReductionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

mlir::LogicalResult IndexReductionPass::processModelOp(ModelOp modelOp) {
  return mlir::success();
}

void IndexReductionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;
  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto handleModel = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");

    return mlir::success();
  };

  if (mlir::failed(mlir::failableParallelForEach(&getContext(), modelOps,
                                                 handleModel))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createIndexReductionPass() {
  return std::make_unique<IndexReductionPass>();
}
} // namespace mlir::bmodelica