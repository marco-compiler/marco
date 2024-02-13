#include "marco/Codegen/Transforms/BindingEquationConversion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_BINDINGEQUATIONCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class BindingEquationConversionPass
      : public mlir::modelica::impl::BindingEquationConversionPassBase<
            BindingEquationConversionPass>
  {
    public:
      using BindingEquationConversionPassBase<BindingEquationConversionPass>
          ::BindingEquationConversionPassBase;

      void runOnOperation() override;
  };
}

namespace
{
  class BindingEquationOpPattern
      : public mlir::OpRewritePattern<BindingEquationOp>
  {
    public:
      BindingEquationOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTable& symbolTable)
          : mlir::OpRewritePattern<BindingEquationOp>(context),
            symbolTable(&symbolTable)
      {
      }

    protected:
      [[nodiscard]] mlir::SymbolTable& getSymbolTable() const
      {
        assert(symbolTable != nullptr);
        return *symbolTable;
      }

    private:
      mlir::SymbolTable* symbolTable;
  };

  struct BindingEquationOpToEquationOpPattern : public BindingEquationOpPattern
  {
    using BindingEquationOpPattern::BindingEquationOpPattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto variable = getSymbolTable().lookup<VariableOp>(op.getVariable());
      return mlir::LogicalResult::success(!variable.isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto modelOp = op->getParentOfType<ModelOp>();

      auto variableOp = getSymbolTable().lookup<VariableOp>(op.getVariable());
      int64_t variableRank = variableOp.getVariableType().getRank();

      // Create the equation template.
      auto templateOp = rewriter.create<EquationTemplateOp>(loc);
      mlir::Block* templateBody = templateOp.createBody(variableRank);
      rewriter.setInsertionPointToStart(templateBody);

      rewriter.mergeBlocks(&op.getBodyRegion().front(), templateBody);

      auto yieldOp = mlir::cast<YieldOp>(templateBody->getTerminator());
      assert(yieldOp.getValues().size() == 1);
      mlir::Value expression = yieldOp.getValues()[0];

      rewriter.setInsertionPointToStart(templateBody);

      mlir::Value lhsValue = rewriter.create<VariableGetOp>(loc, variableOp);
      mlir::Value rhsValue = expression;

      rewriter.setInsertionPointAfter(yieldOp);

      if (auto inductions = templateOp.getInductionVariables();
          !inductions.empty()) {
        lhsValue = rewriter.create<LoadOp>(lhsValue.getLoc(), lhsValue, inductions);
        rhsValue = rewriter.create<LoadOp>(rhsValue.getLoc(), rhsValue, inductions);
      }

      mlir::Value lhsTuple = rewriter.create<EquationSideOp>(loc, lhsValue);
      mlir::Value rhsTuple = rewriter.create<EquationSideOp>(loc, rhsValue);
      rewriter.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

      rewriter.eraseOp(yieldOp);

      rewriter.eraseOp(op);

      // Create the container for the equations.
      rewriter.setInsertionPointAfter(templateOp);

      auto mainModelOp = rewriter.create<MainModelOp>(modelOp.getLoc());
      rewriter.createBlock(&mainModelOp.getBodyRegion());
      rewriter.setInsertionPointToStart(mainModelOp.getBody());

      if (auto indices = variableOp.getIndices(); !indices.empty()) {
        for (const MultidimensionalRange& range : llvm::make_range(
                 indices.rangesBegin(), indices.rangesEnd())) {
          auto instanceOp =
              rewriter.create<EquationInstanceOp>(loc, templateOp);

          instanceOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(rewriter.getContext(), range));
        }
      } else {
        rewriter.create<EquationInstanceOp>(loc, templateOp);
      }
    }
  };

  struct BindingEquationOpToStartOpPattern : public BindingEquationOpPattern
  {
    using BindingEquationOpPattern::BindingEquationOpPattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto variable = getSymbolTable().lookup<VariableOp>(op.getVariable());
      return mlir::LogicalResult::success(variable.isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      auto startOp = rewriter.replaceOpWithNewOp<StartOp>(
          op, mlir::SymbolRefAttr::get(op.getVariableAttr()), true, false);

      assert(startOp.getBodyRegion().empty());

      rewriter.inlineRegionBefore(
          op.getBodyRegion(),
          startOp.getBodyRegion(),
          startOp.getBodyRegion().end());
    }
  };
}

void BindingEquationConversionPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::SymbolTable symbolTable(modelOp);

  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<BindingEquationOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<
      BindingEquationOpToEquationOpPattern,
      BindingEquationOpToStartOpPattern>(&getContext(), symbolTable);

  if (mlir::failed(applyPartialConversion(
          modelOp, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createBindingEquationConversionPass()
  {
    return std::make_unique<BindingEquationConversionPass>();
  }
}
