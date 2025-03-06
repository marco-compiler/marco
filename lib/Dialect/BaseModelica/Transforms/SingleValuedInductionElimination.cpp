#include "marco/Dialect/BaseModelica/Transforms/SingleValuedInductionElimination.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SINGLEVALUEDINDUCTIONELIMINATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class SingleValuedInductionEliminationPass
    : public mlir::bmodelica::impl::SingleValuedInductionEliminationPassBase<
          SingleValuedInductionEliminationPass> {
public:
  using SingleValuedInductionEliminationPassBase<
      SingleValuedInductionEliminationPass>::
      SingleValuedInductionEliminationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  processEquation(mlir::RewriterBase &rewriter,
                  mlir::SymbolTableCollection &symbolTableCollection,
                  EquationInstanceOp equation);

  EquationTemplateOp
  createReducedTemplate(mlir::RewriterBase &rewriter,
                        EquationTemplateOp templateOp,
                        const MultidimensionalRange &indices,
                        const llvm::SmallBitVector &singleValuedInductions);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void SingleValuedInductionEliminationPass::runOnOperation() {
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
SingleValuedInductionEliminationPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<EquationInstanceOp> equations;

  modelOp.walk(
      [&](EquationInstanceOp equation) { equations.push_back(equation); });

  for (EquationInstanceOp equation : equations) {
    if (mlir::failed(
            processEquation(rewriter, symbolTableCollection, equation))) {
      return mlir::failure();
    }
  }

  if (mlir::failed(cleanModelOp(modelOp))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SingleValuedInductionEliminationPass::processEquation(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection,
    EquationInstanceOp equation) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  const IndexSet &equationIndices = equation.getProperties().indices;

  if (equationIndices.empty()) {
    return mlir::success();
  }

  IndexSet remainingIndices = equationIndices;

  for (const MultidimensionalRange &range : llvm::make_range(
           equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
    size_t rank = range.rank();
    llvm::SmallBitVector singleValuedInductions(rank, false);

    for (size_t i = 0; i < rank; ++i) {
      if (range[i].size() == 1) {
        singleValuedInductions[i] = true;
      }
    }

    if (singleValuedInductions.any()) {
      EquationTemplateOp reducedTemplate = createReducedTemplate(
          rewriter, equation.getTemplate(), range, singleValuedInductions);

      if (!reducedTemplate) {
        return mlir::failure();
      }

      rewriter.setInsertionPoint(equation);

      auto newInstance = rewriter.create<EquationInstanceOp>(equation.getLoc(),
                                                             reducedTemplate);

      newInstance.getProperties() = equation.getProperties();

      // Preserve the attributes.
      newInstance->setAttrs(equation->getAttrs());

      // But remove the single-valued dimensions from the indices.
      llvm::SmallVector<Range> newRanges;
      llvm::SmallVector<Range> removedRanges;

      for (size_t i = 0; i < rank; ++i) {
        if (singleValuedInductions[i]) {
          removedRanges.push_back(
              Range(range[i].getBegin(), range[i].getBegin() + 1));
        } else {
          newRanges.push_back(range[i]);
          removedRanges.push_back(range[i]);
        }
      }

      remainingIndices -= MultidimensionalRange(removedRanges);

      if (newRanges.empty()) {
        if (mlir::failed(newInstance.setIndices({}, symbolTableCollection))) {
          return mlir::failure();
        }
      } else {
        if (mlir::failed(newInstance.setIndices(
                IndexSet(MultidimensionalRange(newRanges)),
                symbolTableCollection))) {
          return mlir::failure();
        }
      }
    }
  }

  if (remainingIndices.empty()) {
    // Erase the original instance.
    rewriter.eraseOp(equation);
  } else {
    if (mlir::failed(
            equation.setIndices(remainingIndices, symbolTableCollection))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

EquationTemplateOp SingleValuedInductionEliminationPass::createReducedTemplate(
    mlir::RewriterBase &rewriter,
    mlir::bmodelica::EquationTemplateOp templateOp,
    const MultidimensionalRange &indices,
    const llvm::SmallBitVector &singleValuedInductions) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(templateOp);

  auto reducedTemplateOp =
      rewriter.create<EquationTemplateOp>(templateOp.getLoc());

  // Preserve the attributes of the original template.
  reducedTemplateOp->setAttrs(templateOp->getAttrs());

  mlir::Block *bodyBlock =
      reducedTemplateOp.createBody(templateOp.getInductionVariables().size() -
                                   singleValuedInductions.count());

  rewriter.setInsertionPointToStart(bodyBlock);
  mlir::IRMapping mapping;
  size_t pos = 0;

  for (size_t i = 0, rank = indices.rank(); i < rank; ++i) {
    if (singleValuedInductions[i]) {
      mlir::Value constantValue = rewriter.create<ConstantOp>(
          templateOp.getLoc(), rewriter.getIndexAttr(indices[i].getBegin()));

      mapping.map(templateOp.getInductionVariables()[i], constantValue);
    } else {
      mapping.map(templateOp.getInductionVariables()[i],
                  reducedTemplateOp.getInductionVariables()[pos++]);
    }
  }

  for (auto &op : templateOp.getOps()) {
    rewriter.clone(op, mapping);
  }

  return reducedTemplateOp;
}

mlir::LogicalResult
SingleValuedInductionEliminationPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSingleValuedInductionEliminationPass() {
  return std::make_unique<SingleValuedInductionEliminationPass>();
}
} // namespace mlir::bmodelica
