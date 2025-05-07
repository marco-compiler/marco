#include "marco/Dialect/BaseModelica/Transforms/ScalarRangesEquationSplit.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCALARRANGESEQUATIONSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class ScalarRangesEquationSplitPass
    : public impl::ScalarRangesEquationSplitPassBase<
          ScalarRangesEquationSplitPass> {
public:
  using ScalarRangesEquationSplitPassBase<
      ScalarRangesEquationSplitPass>::ScalarRangesEquationSplitPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

namespace {
mlir::LogicalResult splitEquation(EquationInstanceOp equationOp,
                                  mlir::SymbolTableCollection &symbolTables) {
  mlir::IRRewriter rewriter(equationOp);
  const IndexSet &indices = equationOp.getProperties().indices;

  if (indices.empty()) {
    // Nothing to do in case of scalar equation.
    return mlir::success();
  }

  llvm::SmallVector<MultidimensionalRange> remaining;
  bool modified = false;

  for (const MultidimensionalRange &range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    if (range.flatSize() == 1) {
      auto clonedOp = mlir::cast<EquationInstanceOp>(
          rewriter.clone(*equationOp.getOperation()));

      if (mlir::failed(clonedOp.setIndices(IndexSet(range), symbolTables))) {
        return mlir::failure();
      }

      modified = true;
    } else {
      remaining.push_back(range);
    }
  }

  if (modified) {
    if (remaining.empty()) {
      rewriter.eraseOp(equationOp);
    } else {
      if (mlir::failed(
              equationOp.setIndices(IndexSet(remaining), symbolTables))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}
} // namespace

void ScalarRangesEquationSplitPass::runOnOperation() {
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

mlir::LogicalResult
ScalarRangesEquationSplitPass::processModelOp(ModelOp modelOp) {
  llvm::SmallVector<EquationInstanceOp> equationOps;

  modelOp.walk([&](EquationInstanceOp equationOp) {
    equationOps.push_back(equationOp);
  });

  mlir::SymbolTableCollection symbolTables;

  for (EquationInstanceOp equationOp : equationOps) {
    if (mlir::failed(splitEquation(equationOp, symbolTables))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createScalarRangesEquationSplitPass() {
  return std::make_unique<ScalarRangesEquationSplitPass>();
}
} // namespace mlir::bmodelica
