#include "marco/Dialect/BaseModelica/Transforms/CallCSE.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_CALLCSEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class CallCSEPass : public impl::CallCSEPassBase<CallCSEPass> {
public:
  using CallCSEPassBase<CallCSEPass>::CallCSEPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

void CallCSEPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), modelOps, [&](mlir::Operation *op) {
            return processModelOp(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult CallCSEPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(modelOp);
  llvm::SmallVector<EquationInstanceOp> initialEquationOps;
  llvm::SmallVector<EquationInstanceOp> dynamicEquationOps;

  modelOp.collectInitialEquations(initialEquationOps);
  modelOp.collectMainEquations(dynamicEquationOps);

  llvm::DenseSet<EquationTemplateOp> templateOps;

  for (EquationInstanceOp equationOp : initialEquationOps) {
    templateOps.insert(equationOp.getTemplate());
  }

  for (EquationInstanceOp equationOp : dynamicEquationOps) {
    templateOps.insert(equationOp.getTemplate());
  }

  llvm::SmallVector<CallOp> callOps;

  for (EquationTemplateOp templateOp : templateOps) {
    templateOp->walk([&](CallOp callOp) {
      callOps.push_back(callOp);
    });
  }

  

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createCallCSEPass() {
  return std::make_unique<CallCSEPass>();
}
} // namespace mlir::bmodelica
