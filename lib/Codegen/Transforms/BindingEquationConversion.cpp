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
      mlir::SymbolTable& getSymbolTable() const
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

      auto yieldOp = mlir::cast<YieldOp>(
          op.getBodyRegion().back().getTerminator());

      assert(yieldOp.getValues().size() == 1);
      mlir::Value expression = yieldOp.getValues()[0];

      auto variableOp = getSymbolTable().lookup<VariableOp>(op.getVariable());

      // Create the equation template.
      auto equationTemplateOp = rewriter.create<EquationTemplateOp>(loc);

      rewriter.setInsertionPointToStart(&op.getBodyRegion().front());

      mlir::Value lhsValue = rewriter.create<VariableGetOp>(loc, variableOp);
      mlir::Value rhsValue = expression;

      rewriter.setInsertionPoint(yieldOp);

      mlir::Value lhsTuple = rewriter.create<EquationSideOp>(loc, lhsValue);
      mlir::Value rhsTuple = rewriter.create<EquationSideOp>(loc, rhsValue);
      rewriter.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

      rewriter.eraseOp(yieldOp);

      rewriter.inlineRegionBefore(
          op.getBodyRegion(),
          equationTemplateOp.getBodyRegion(),
          equationTemplateOp.getBodyRegion().end());

      rewriter.eraseOp(op);

      // Create the equation instance.
      rewriter.setInsertionPointAfter(equationTemplateOp);
      rewriter.create<EquationInstanceOp>(loc, equationTemplateOp, false);
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
          op, op.getVariable(), true, false);

      assert(startOp.getBodyRegion().empty());

      rewriter.inlineRegionBefore(
          op.getBodyRegion(),
          startOp.getBodyRegion(),
          startOp.getBodyRegion().end());
    }
  };
}

namespace
{
  class BindingEquationConversionPass
      : public mlir::modelica::impl::BindingEquationConversionPassBase<
            BindingEquationConversionPass>
  {
    public:
      using BindingEquationConversionPassBase
        ::BindingEquationConversionPassBase;

      void runOnOperation() override
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
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createBindingEquationConversionPass()
  {
    return std::make_unique<BindingEquationConversionPass>();
  }
}
