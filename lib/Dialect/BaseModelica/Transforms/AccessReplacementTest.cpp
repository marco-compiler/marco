#include "marco/Dialect/BaseModelica/Transforms/AccessReplacementTest.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_ACCESSREPLACEMENTTESTPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

using EquationsMap = llvm::StringMap<EquationInstanceOp>;

namespace {
class AccessReplacePattern : public mlir::OpRewritePattern<EquationInstanceOp> {
public:
  AccessReplacePattern(mlir::MLIRContext *context,
                       mlir::SymbolTableCollection &symbolTableCollection,
                       const EquationsMap &equationsMap)
      : mlir::OpRewritePattern<EquationInstanceOp>(context),
        symbolTableCollection(&symbolTableCollection),
        equationsMap(&equationsMap) {}

  mlir::LogicalResult
  matchAndRewrite(EquationInstanceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Obtain the information about the access to be replaced.
    IndexSet indices;

    if (auto indicesAttr =
            op->getAttrOfType<MultidimensionalRangeAttr>("replace_indices")) {
      indices = IndexSet(indicesAttr.getValue());
    }

    auto destinationPath =
        op->getAttrOfType<EquationPathAttr>("replace_destination_path");

    auto replacementEquationId =
        op->getAttrOfType<mlir::StringAttr>("replace_eq");

    auto sourcePath =
        op->getAttrOfType<EquationPathAttr>("replace_source_path");

    if (!destinationPath || !replacementEquationId || !sourcePath) {
      return mlir::failure();
    }

    std::optional<VariableAccess> destinationAccess =
        op.getAccessAtPath(*symbolTableCollection, destinationPath.getValue());

    if (!destinationAccess) {
      return mlir::failure();
    }

    EquationInstanceOp replacementEquation =
        equationsMap->lookup(replacementEquationId.getValue());

    if (!replacementEquation) {
      return mlir::failure();
    }

    std::optional<VariableAccess> sourceAccess =
        replacementEquation.getAccessAtPath(*symbolTableCollection,
                                            sourcePath.getValue());

    if (!sourceAccess) {
      return mlir::failure();
    }

    // Remove the attributes before cloning, in order to avoid infinite
    // loops.

    op->removeAttr("replace_indices");
    op->removeAttr("replace_destination_path");
    op->removeAttr("replace_eq");
    op->removeAttr("replace_source_path");

    // Perform the replacement.
    llvm::SmallVector<EquationInstanceOp> clones;

    if (mlir::failed(op.cloneWithReplacedAccess(
            rewriter, *symbolTableCollection,
            std::reference_wrapper<const IndexSet>(indices), *destinationAccess,
            replacementEquation.getTemplate(), *sourceAccess, clones))) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
  const EquationsMap *equationsMap;
};
} // namespace

namespace {
class AccessReplacementTestPass
    : public mlir::bmodelica::impl::AccessReplacementTestPassBase<
          AccessReplacementTestPass> {
public:
  using AccessReplacementTestPassBase::AccessReplacementTestPassBase;

  void runOnOperation() override {
    llvm::DebugFlag = true;
    ModelOp modelOp = getOperation();

    llvm::SmallVector<EquationInstanceOp> equationOps;
    EquationsMap equationsMap;

    modelOp->walk([&](EquationInstanceOp equationOp) {
      equationOps.push_back(equationOp);

      if (auto idAttr = equationOp->getAttrOfType<mlir::StringAttr>("id")) {
        equationsMap[idAttr.getValue()] = equationOp;
      }
    });

    mlir::RewritePatternSet patterns(&getContext());
    mlir::SymbolTableCollection symbolTableCollection;

    patterns.add<AccessReplacePattern>(&getContext(), symbolTableCollection,
                                       equationsMap);

    mlir::GreedyRewriteConfig config;
    config.fold = true;

    if (mlir::failed(mlir::applyPatternsGreedily(modelOp, std::move(patterns),
                                                 config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createAccessReplacementTestPass() {
  return std::make_unique<AccessReplacementTestPass>();
}
} // namespace mlir::bmodelica
