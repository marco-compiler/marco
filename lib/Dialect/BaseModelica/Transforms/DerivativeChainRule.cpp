#include "marco/Dialect/BaseModelica/Transforms/DerivativeChainRule.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_DERIVATIVECHAINRULEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class DerivativeChainRulePass
      : public impl::DerivativeChainRulePassBase<DerivativeChainRulePass>
  {
    public:
      using DerivativeChainRulePassBase<DerivativeChainRulePass>
          ::DerivativeChainRulePassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult applyChainRule();
  };
}

void DerivativeChainRulePass::runOnOperation()
{
  if (mlir::failed(applyChainRule())) {
    return signalPassFailure();
  }
}

namespace
{
  class FoldDerPattern : public mlir::OpRewritePattern<DerOp>
  {
    public:
      FoldDerPattern(
          mlir::MLIRContext* context,
          llvm::DenseMap<mlir::Operation*, ad::forward::State>& states)
          : mlir::OpRewritePattern<DerOp>::OpRewritePattern(context),
            states(&states)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          DerOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Operation* classOp =
            op->getParentWithTrait<ClassInterface::Trait>();

        if (!mlir::isa<FunctionOp>(classOp)) {
          return mlir::failure();
        }

        // Check if the argument can be derived.
        auto operandOp = op.getOperand().getDefiningOp<DerivableOpInterface>();

        if (!operandOp) {
          return mlir::failure();
        }

        // Map the variables inside the class.
        if (states->find(classOp) == states->end()) {
          auto& state = (*states)[classOp];

          if (mlir::failed(mapClassVariables(classOp, state))) {
            op.emitOpError()
                << "Can't map derivative variables of parent class";

            return mlir::failure();
          }
        }

        // Compute the derivative.
        auto& state = (*states)[classOp];

        if (mlir::failed(operandOp.createTimeDerivative(
                rewriter, state, true))) {
          return mlir::failure();
        }

        auto derivative = state.getDerivative(op.getOperand());

        if (!derivative) {
          return mlir::failure();
        }

        rewriter.replaceOp(op, *derivative);
        return mlir::success();
      }

    private:
      mlir::LogicalResult mapClassVariables(
          mlir::Operation* op, ad::forward::State& state) const
      {
        if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
          mapTimeDerivativeFunctionVariables(functionOp, state);
          return mlir::success();
        }

        return mlir::failure();
      }

    private:
      llvm::DenseMap<mlir::Operation*, ad::forward::State>* states;
  };
}

mlir::LogicalResult DerivativeChainRulePass::applyChainRule()
{
  ad::forward::State state;
  llvm::DenseMap<mlir::Operation*, ad::forward::State> states;

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<FoldDerPattern>(&getContext(), states);

  return mlir::applyPatternsAndFoldGreedily(
      getOperation(), std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createDerivativeChainRulePass()
  {
    return std::make_unique<DerivativeChainRulePass>();
  }
}
