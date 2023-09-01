#include "marco/Codegen/Transforms/EquationExplicitationTest.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ExplicitatePattern
      : public mlir::OpRewritePattern<MatchedEquationInstanceOp>
  {
    public:
      ExplicitatePattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::DenseSet<MatchedEquationInstanceOp>& processedEquations)
          : mlir::OpRewritePattern<MatchedEquationInstanceOp>(context),
            symbolTableCollection(&symbolTableCollection),
            processedEquations(&processedEquations)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MatchedEquationInstanceOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        if (processedEquations->contains(op)) {
          return mlir::failure();
        }

        processedEquations->insert(op);
        return op.explicitate(rewriter, *symbolTableCollection);
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      llvm::DenseSet<MatchedEquationInstanceOp>* processedEquations;
  };
}

namespace
{
  class EquationExplicitationTestPass
      : public impl::EquationExplicitationTestPassBase<
            EquationExplicitationTestPass>
  {
    public:
      using EquationExplicitationTestPassBase
          ::EquationExplicitationTestPassBase;

      void runOnOperation() override;
  };
}

void EquationExplicitationTestPass::runOnOperation()
{
  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::DenseSet<MatchedEquationInstanceOp> processedEquations;

  patterns.add<ExplicitatePattern>(
      &getContext(), symbolTableCollection, processedEquations);

  if (mlir::failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationExplicitationTestPass()
  {
    return std::make_unique<EquationExplicitationTestPass>();
  }
}
