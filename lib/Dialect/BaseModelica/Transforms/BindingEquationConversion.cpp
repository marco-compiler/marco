#include "marco/Dialect/BaseModelica/Transforms/BindingEquationConversion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_BINDINGEQUATIONCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class BindingEquationConversionPass
    : public mlir::bmodelica::impl::BindingEquationConversionPassBase<
          BindingEquationConversionPass> {
public:
  using BindingEquationConversionPassBase<
      BindingEquationConversionPass>::BindingEquationConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

namespace {
class BindingEquationOpPattern
    : public mlir::OpRewritePattern<BindingEquationOp> {
public:
  BindingEquationOpPattern(mlir::MLIRContext *context,
                           mlir::SymbolTable &symbolTable)
      : mlir::OpRewritePattern<BindingEquationOp>(context),
        symbolTable(&symbolTable) {}

protected:
  [[nodiscard]] mlir::SymbolTable &getSymbolTable() const {
    assert(symbolTable != nullptr);
    return *symbolTable;
  }

private:
  mlir::SymbolTable *symbolTable;
};

struct BindingEquationOpToEquationOpPattern : public BindingEquationOpPattern {
  using BindingEquationOpPattern::BindingEquationOpPattern;

  mlir::LogicalResult match(BindingEquationOp op) const override {
    auto variable = getSymbolTable().lookup<VariableOp>(op.getVariable());
    return mlir::LogicalResult::success(!variable.isReadOnly());
  }

  void rewrite(BindingEquationOp op,
               mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto modelOp = op->getParentOfType<ModelOp>();

    auto variableOp = getSymbolTable().lookup<VariableOp>(op.getVariable());
    int64_t variableRank = variableOp.getVariableType().getRank();

    // Create the equation template.
    auto templateOp = rewriter.create<EquationTemplateOp>(loc);
    mlir::Block *templateBody = templateOp.createBody(variableRank);
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
      lhsValue = rewriter.create<TensorExtractOp>(lhsValue.getLoc(), lhsValue,
                                                  inductions);

      rhsValue = rewriter.create<TensorExtractOp>(rhsValue.getLoc(), rhsValue,
                                                  inductions);
    }

    mlir::Value lhsTuple = rewriter.create<EquationSideOp>(loc, lhsValue);
    mlir::Value rhsTuple = rewriter.create<EquationSideOp>(loc, rhsValue);
    rewriter.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

    rewriter.eraseOp(yieldOp);

    rewriter.eraseOp(op);

    // Create the container for the equations.
    rewriter.setInsertionPointAfter(templateOp);

    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());

    auto instanceOp = rewriter.create<EquationInstanceOp>(loc, templateOp);
    instanceOp.getProperties().setIndices(variableOp.getIndices());
  }
};

struct BindingEquationOpToStartOpPattern : public BindingEquationOpPattern {
  using BindingEquationOpPattern::BindingEquationOpPattern;

  mlir::LogicalResult match(BindingEquationOp op) const override {
    auto variable = getSymbolTable().lookup<VariableOp>(op.getVariable());
    return mlir::LogicalResult::success(variable.isReadOnly());
  }

  void rewrite(BindingEquationOp op,
               mlir::PatternRewriter &rewriter) const override {
    auto startOp = rewriter.replaceOpWithNewOp<StartOp>(
        op, mlir::SymbolRefAttr::get(op.getVariableAttr()), true, false);

    assert(startOp.getBodyRegion().empty());

    rewriter.inlineRegionBefore(op.getBodyRegion(), startOp.getBodyRegion(),
                                startOp.getBodyRegion().end());
  }
};
} // namespace

void BindingEquationConversionPass::runOnOperation() {
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
BindingEquationConversionPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTable symbolTable(modelOp);

  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<BindingEquationOp>();

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<BindingEquationOpToEquationOpPattern,
                  BindingEquationOpToStartOpPattern>(&getContext(),
                                                     symbolTable);

  return applyPartialConversion(modelOp, target, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createBindingEquationConversionPass() {
  return std::make_unique<BindingEquationConversionPass>();
}
} // namespace mlir::bmodelica
