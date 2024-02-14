#include "marco/Codegen/Transforms/SingleValuedInductionElimination.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SINGLEVALUEDINDUCTIONELIMINATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class SingleValuedInductionEliminationPass
      : public mlir::modelica::impl::SingleValuedInductionEliminationPassBase<
            SingleValuedInductionEliminationPass>
  {
    public:
      using SingleValuedInductionEliminationPassBase<
          SingleValuedInductionEliminationPass>::
        SingleValuedInductionEliminationPassBase;

      void runOnOperation() override;

      private:
        mlir::LogicalResult processEquation(
            mlir::RewriterBase& rewriter,
            MatchedEquationInstanceOp equation);

        EquationTemplateOp createReducedTemplate(
            mlir::RewriterBase& rewriter,
            EquationTemplateOp templateOp,
            const MultidimensionalRange& indices,
            const llvm::SmallBitVector& singleValuedInductions);

        mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void SingleValuedInductionEliminationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<MatchedEquationInstanceOp> equations;

  modelOp.walk([&](MatchedEquationInstanceOp equation) {
    equations.push_back(equation);
  });

  for (MatchedEquationInstanceOp equation : equations) {
    if (mlir::failed(processEquation(rewriter, equation))) {
      return signalPassFailure();
    }
  }

  if (mlir::failed(cleanModelOp(modelOp))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult SingleValuedInductionEliminationPass::processEquation(
    mlir::RewriterBase& rewriter,
    MatchedEquationInstanceOp equation)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  if (auto indices = equation.getIndices()) {
    size_t rank = indices->getValue().rank();
    llvm::SmallBitVector singleValuedInductions(rank, false);

    for (size_t i = 0; i < rank; ++i) {
      if (indices->getValue()[i].size() == 1) {
        singleValuedInductions[i] = true;
      }
    }

    if (singleValuedInductions.any()) {
      EquationTemplateOp reducedTemplate = createReducedTemplate(
          rewriter, equation.getTemplate(), indices->getValue(),
          singleValuedInductions);

      if (!reducedTemplate) {
        return mlir::failure();
      }

      rewriter.setInsertionPoint(equation);

      auto newInstance = rewriter.create<MatchedEquationInstanceOp>(
          equation.getLoc(), reducedTemplate, equation.getPath());

      // Preserve the attributes.
      newInstance->setAttrs(equation->getAttrs());

      // But remove the single-valued dimensions from the indices.
      llvm::SmallVector<Range> newRanges;

      for (size_t i = 0; i < rank; ++i) {
        if (!singleValuedInductions[i]) {
          newRanges.push_back(indices->getValue()[i]);
        }
      }

      if (newRanges.empty()) {
        newInstance.removeIndicesAttr();
      } else {
        newInstance.setIndicesAttr(MultidimensionalRangeAttr::get(
            rewriter.getContext(), MultidimensionalRange(newRanges)));
      }

      // Erase the original instance.
      rewriter.eraseOp(equation);
    }
  }

  return mlir::success();
}

EquationTemplateOp
SingleValuedInductionEliminationPass::createReducedTemplate(
    mlir::RewriterBase& rewriter,
    mlir::modelica::EquationTemplateOp templateOp,
    const MultidimensionalRange& indices,
    const llvm::SmallBitVector& singleValuedInductions)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(templateOp);

  auto reducedTemplateOp =
      rewriter.create<EquationTemplateOp>(templateOp.getLoc());

  // Preserve the attributes of the original template.
  reducedTemplateOp->setAttrs(templateOp->getAttrs());

  mlir::Block* bodyBlock = reducedTemplateOp.createBody(
      templateOp.getInductionVariables().size() -
      singleValuedInductions.count());

  rewriter.setInsertionPointToStart(bodyBlock);
  mlir::IRMapping mapping;
  size_t pos = 0;

  for (size_t i = 0, rank = indices.rank(); i < rank; ++i) {
    if (singleValuedInductions[i]) {
      mlir::Value constantValue = rewriter.create<ConstantOp>(
          templateOp.getLoc(),
          rewriter.getIndexAttr(indices[i].getBegin()));

      mapping.map(templateOp.getInductionVariables()[i], constantValue);
    } else {
      mapping.map(templateOp.getInductionVariables()[i],
                  reducedTemplateOp.getInductionVariables()[pos++]);
    }
  }

  for (auto& op : templateOp.getOps()) {
    rewriter.clone(op, mapping);
  }

  return reducedTemplateOp;
}

mlir::LogicalResult
SingleValuedInductionEliminationPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSingleValuedInductionEliminationPass()
  {
    return std::make_unique<SingleValuedInductionEliminationPass>();
  }
}
