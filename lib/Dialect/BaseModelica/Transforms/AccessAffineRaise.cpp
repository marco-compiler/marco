#include "marco/Dialect/BaseModelica/Transforms/AccessAffineRaise.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_ACCESSAFFINERAISEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class AccessAffineRaisePass
    : public impl::AccessAffineRaisePassBase<AccessAffineRaisePass> {
public:
  using AccessAffineRaisePassBase::AccessAffineRaisePassBase;

  void runOnOperation() override;

private:
  void processMemoryOp(mlir::Operation *op, mlir::RewriterBase &rewriter);
};
} // namespace

namespace {
template <typename Op>
class RaiseSubscriptsPattern {
public:
  virtual void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                             Op op) const = 0;

  virtual void
  replaceOp(mlir::RewriterBase &rewriter, Op op,
            llvm::ArrayRef<mlir::Value> affineSubscripts) const = 0;

  void run(Op op, mlir::RewriterBase &rewriter) const {
    // Collect the induction variables.
    llvm::SmallVector<mlir::Value> inductions;
    mlir::Operation *parentOp = op->getParentOp();

    while (parentOp && isAffineLoop(parentOp)) {
      if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(parentOp)) {
        inductions.push_back(forOp.getInductionVar());
      } else if (auto parallelForOp =
                     mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
        auto inductionVars = parallelForOp.getIVs();
        inductions.append(inductionVars.rbegin(), inductionVars.rend());
      }

      parentOp = parentOp->getParentOp();
    }

    std::reverse(inductions.begin(), inductions.end());

    // Map the positions of the induction variables.
    llvm::DenseMap<mlir::Value, int64_t> inductionsPosMap;

    for (auto induction : llvm::enumerate(inductions)) {
      inductionsPosMap[induction.value()] =
          static_cast<int64_t>(induction.index());
    }

    // Try to get the affine expressions for the subscripts.
    llvm::SmallVector<mlir::Value> subscripts;
    getSubscripts(subscripts, op);

    if (subscripts.empty()) {
      return;
    }

    llvm::SmallVector<mlir::AffineMap> affineMaps;

    for (mlir::Value subscript : subscripts) {
      if (subscript.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        return;
      }
      auto affineSubscriptExp = getAffineExpr(subscript, inductionsPosMap);

      if (!affineSubscriptExp) {
        return;
      }

      affineMaps.push_back(
          mlir::AffineMap::get(inductions.size(), 0, *affineSubscriptExp));
    }

    // Apply the affine expressions to the induction variables.
    llvm::SmallVector<mlir::Value> affineSubscripts;

    for (auto affineMap : llvm::enumerate(affineMaps)) {
      mlir::Value affineSubscript =
          rewriter.create<mlir::affine::AffineApplyOp>(
              subscripts[affineMap.index()].getLoc(), affineMap.value(),
              inductions);

      affineSubscripts.push_back(affineSubscript);
    }

    replaceOp(rewriter, op, affineSubscripts);
  }

private:
  bool isAffineLoop(mlir::Operation *op) const {
    return mlir::isa<mlir::affine::AffineForOp, mlir::affine::AffineParallelOp>(
        op);
  }

  std::optional<mlir::AffineExpr> getAffineExpr(
      mlir::Value subscript,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    if (inductionsPosMap.contains(subscript)) {
      return mlir::getAffineDimExpr(inductionsPosMap.lookup(subscript),
                                    subscript.getContext());
    }

    auto affineExpInt = subscript.getDefiningOp<AffineLikeOpInterface>();

    if (!affineExpInt) {
      return std::nullopt;
    }

    return affineExpInt.getAffineExpression(inductionsPosMap);
  }
};

class LoadOpPattern : public RaiseSubscriptsPattern<mlir::memref::LoadOp> {
public:
  using RaiseSubscriptsPattern<mlir::memref::LoadOp>::RaiseSubscriptsPattern;

  void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                     mlir::memref::LoadOp op) const override {
    llvm::append_range(subscripts, op.getIndices());
  }

  void replaceOp(mlir::RewriterBase &rewriter, mlir::memref::LoadOp op,
                 llvm::ArrayRef<mlir::Value> affineSubscripts) const override {
    rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(op, op.getMemRef(),
                                                            affineSubscripts);
  }
};

class StoreOpPattern : public RaiseSubscriptsPattern<mlir::memref::StoreOp> {
public:
  using RaiseSubscriptsPattern<mlir::memref::StoreOp>::RaiseSubscriptsPattern;

  void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                     mlir::memref::StoreOp op) const override {
    llvm::append_range(subscripts, op.getIndices());
  }

  void replaceOp(mlir::RewriterBase &rewriter, mlir::memref::StoreOp op,
                 llvm::ArrayRef<mlir::Value> affineSubscripts) const override {
    rewriter.replaceOpWithNewOp<mlir::affine::AffineStoreOp>(
        op, op.getValue(), op.getMemRef(), affineSubscripts);
  }
};
} // namespace

void AccessAffineRaisePass::runOnOperation() {
  llvm::SmallVector<mlir::Operation *> memoryOps;

  getOperation().walk([&](mlir::Operation *nestedOp) {
    if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(nestedOp)) {
      memoryOps.push_back(nestedOp);
    }
  });

  mlir::IRRewriter rewriter(&getContext());

  for (mlir::Operation *memoryOp : memoryOps) {
    processMemoryOp(memoryOp, rewriter);
  }
}

void AccessAffineRaisePass::processMemoryOp(mlir::Operation *op,
                                            mlir::RewriterBase &rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
    LoadOpPattern pattern;
    pattern.run(loadOp, rewriter);
  } else if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    StoreOpPattern pattern;
    pattern.run(storeOp, rewriter);
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createAccessAffineRaisePass() {
  return std::make_unique<AccessAffineRaisePass>();
}
} // namespace mlir::bmodelica
