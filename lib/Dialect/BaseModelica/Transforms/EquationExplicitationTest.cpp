#include "marco/Dialect/BaseModelica/Transforms/EquationExplicitationTest.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONTESTPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class ExplicitatePattern : public mlir::OpRewritePattern<EquationInstanceOp> {
public:
  ExplicitatePattern(mlir::MLIRContext *context,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     llvm::DenseSet<EquationInstanceOp> &processedEquations)
      : mlir::OpRewritePattern<EquationInstanceOp>(context),
        symbolTableCollection(&symbolTableCollection),
        processedEquations(&processedEquations) {}

  mlir::LogicalResult
  matchAndRewrite(EquationInstanceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (processedEquations->contains(op)) {
      return mlir::failure();
    }

    processedEquations->insert(op);
    return op.explicitate(rewriter, *symbolTableCollection);
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
  llvm::DenseSet<EquationInstanceOp> *processedEquations;
};
} // namespace

namespace {
class EquationExplicitationTestPass
    : public impl::EquationExplicitationTestPassBase<
          EquationExplicitationTestPass> {
public:
  using EquationExplicitationTestPassBase ::EquationExplicitationTestPassBase;

  void runOnOperation() override;
};
} // namespace

void EquationExplicitationTestPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::DenseSet<EquationInstanceOp> processedEquations;

  patterns.add<ExplicitatePattern>(&getContext(), symbolTableCollection,
                                   processedEquations);

  mlir::GreedyRewriteConfig config;
  config.enableFolding();

  if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                               std::move(patterns), config))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationExplicitationTestPass() {
  return std::make_unique<EquationExplicitationTestPass>();
}
} // namespace mlir::bmodelica
