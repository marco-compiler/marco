#include "marco/Codegen/Transforms/EquationExplicitation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class EquationExplicitationPass
      : public impl::EquationExplicitationPassBase<
            EquationExplicitationPass>
  {
    public:
      using EquationExplicitationPassBase<EquationExplicitationPass>
          ::EquationExplicitationPassBase;

      void runOnOperation() override;

    private:
      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          ScheduledEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult processScheduleOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduleOp scheduleOp);

      mlir::LogicalResult processSCCs(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult processSCC(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          SCCOp scc);

      EquationFunctionOp createEquationFunction(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equation,
          llvm::SmallVectorImpl<uint64_t>& lowerBounds,
          llvm::SmallVectorImpl<uint64_t>& upperBounds,
          llvm::SmallVectorImpl<uint64_t>& steps);

      llvm::SmallVector<mlir::Value> shiftInductions(
          mlir::RewriterBase& rewriter,
          mlir::ArrayAttr iterationDirections,
          const MultidimensionalRange& indices,
          mlir::ValueRange loopInductions);

      mlir::LogicalResult cloneEquationTemplateIntoFunction(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          EquationTemplateOp templateOp,
          llvm::ArrayRef<mlir::Value> inductions);

      mlir::LogicalResult getAccessAttrs(
          llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
          llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
          mlir::SymbolTableCollection& symbolTableCollection,
          ScheduledEquationInstanceOp equationOp);

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void EquationExplicitationPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp, 1> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    for (ScheduleOp scheduleOp :
         llvm::make_early_inc_range(modelOp.getOps<ScheduleOp>())) {
      if (mlir::failed(processScheduleOp(
              rewriter, symbolTableCollection, moduleOp, modelOp,
              scheduleOp))) {
        return signalPassFailure();
      }
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationExplicitationPass::getVariableAccessAnalysis(
    ScheduledEquationInstanceOp equation,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  auto modelOp = equation->getParentOfType<ModelOp>();
  auto analysisManager = getAnalysisManager().nest(modelOp);

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equation.getTemplate())) {
    return *analysis;
  }

  auto& analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equation.getTemplate());

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult EquationExplicitationPass::processScheduleOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduleOp scheduleOp)
{
  llvm::SmallVector<SCCOp> SCCs;

  for (auto& op : scheduleOp.getOps()) {
    if (auto initialModelOp = mlir::dyn_cast<InitialModelOp>(op)) {
      initialModelOp.collectSCCs(SCCs);
      continue;
    }

    if (auto mainModelOp = mlir::dyn_cast<MainModelOp>(op)) {
      mainModelOp.collectSCCs(SCCs);
      continue;
    }
  }

  return processSCCs(rewriter, symbolTableCollection, moduleOp, modelOp, SCCs);
}

mlir::LogicalResult EquationExplicitationPass::processSCCs(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  for (SCCOp scc : SCCs) {
    if (mlir::failed(processSCC(
            rewriter, symbolTableCollection, moduleOp, modelOp, scc))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCC(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    SCCOp scc)
{
  llvm::SmallVector<ScheduledEquationInstanceOp> equations;
  scc.collectEquations(equations);

  if (equations.size() > 1) {
    // Cycle among the equations.
    return mlir::success();
  }

  bool isSCCErasable = true;

  for (ScheduledEquationInstanceOp equation : equations) {
    ScheduledEquationInstanceOp explicitEquation =
        equation.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (explicitEquation) {
      llvm::SmallVector<uint64_t, 10> lowerBounds;
      llvm::SmallVector<uint64_t, 10> upperBounds;
      llvm::SmallVector<uint64_t, 10> steps;

      EquationFunctionOp eqFunc = createEquationFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, explicitEquation,
          lowerBounds, upperBounds, steps);

      if (!eqFunc) {
        return mlir::failure();
      }

      rewriter.setInsertionPoint(scc);

      llvm::SmallVector<mlir::Attribute> writtenVariables;
      llvm::SmallVector<mlir::Attribute> readVariables;

      if (mlir::failed(getAccessAttrs(
              writtenVariables, readVariables, symbolTableCollection,
              equation))) {
        return mlir::failure();
      }

      auto scheduleBlockOp = rewriter.create<ScheduleBlockOp>(
          modelOp.getLoc(),
          true,
          rewriter.getArrayAttr(writtenVariables),
          rewriter.getArrayAttr(readVariables));

      rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
      rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

      llvm::SmallVector<Range> ranges;

      for (size_t i = 0, e = equation.getInductionVariables().size();
           i < e; ++i) {
        ranges.push_back(Range(
            static_cast<Point::data_type>(lowerBounds[i]),
            static_cast<Point::data_type>(upperBounds[i])));

        assert(steps[i] == 1);
      }

      auto iterationDirections = equation.getIterationDirections();
      bool independentIndices = !iterationDirections.empty();

      if (independentIndices) {
        independentIndices &= llvm::all_of(
            equation.getIterationDirections(),
            [](mlir::Attribute attr) {
              return attr.cast<EquationScheduleDirectionAttr>().getValue() ==
                  EquationScheduleDirection::Any;
            });
      }

      auto callOp = rewriter.create<EquationCallOp>(
          eqFunc.getLoc(), eqFunc.getSymName(), nullptr, independentIndices);

      if (!ranges.empty()) {
        callOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            rewriter.getContext(), MultidimensionalRange(ranges)));
      }

      rewriter.eraseOp(explicitEquation);
    } else {
      isSCCErasable = false;
    }
  }

  if (isSCCErasable) {
    rewriter.eraseOp(scc);
  }

  return mlir::success();
}

EquationFunctionOp EquationExplicitationPass::createEquationFunction(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equation,
    llvm::SmallVectorImpl<uint64_t>& lowerBounds,
    llvm::SmallVectorImpl<uint64_t>& upperBounds,
    llvm::SmallVectorImpl<uint64_t>& steps)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto eqFunc = rewriter.create<EquationFunctionOp>(
      equation.getLoc(), "equation", equation.getInductionVariables().size());

  symbolTableCollection.getSymbolTable(moduleOp).insert(eqFunc);

  mlir::Block* entryBlock = eqFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (auto indicesAttr = equation.getIndices()) {
    size_t rank = indicesAttr->getValue().rank();
    auto iterationDirections = equation.getIterationDirections();

    for (size_t i = 0; i < rank; ++i) {
      auto iterationDirectionAttr =
          iterationDirections[i].cast<EquationScheduleDirectionAttr>();

      lowerBounds.push_back(0);
      steps.push_back(1);

      auto iterationDirection = iterationDirectionAttr.getValue();

      if (iterationDirection == EquationScheduleDirection::Any ||
          iterationDirection == EquationScheduleDirection::Forward) {
        auto upperBound = indicesAttr->getValue()[i].getEnd() -
            indicesAttr->getValue()[i].getBegin();

        upperBounds.push_back(upperBound);
      } else {
        assert(iterationDirection == EquationScheduleDirection::Backward);

        auto upperBound = indicesAttr->getValue()[i].getBegin() -
            indicesAttr->getValue()[i].getEnd();

        upperBounds.push_back(upperBound);
      }
    }

    llvm::SmallVector<mlir::Value> inductions;

    mlir::Value oneValue = rewriter.create<mlir::arith::ConstantOp>(
        equation.getLoc(), rewriter.getIndexAttr(1));

    for (size_t i = 0; i < rank; ++i) {
      auto forOp = rewriter.create<mlir::scf::ForOp>(
          equation.getLoc(),
          eqFunc.getLowerBound(i),
          eqFunc.getUpperBound(i),
          oneValue);

      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    auto shiftedInductions = shiftInductions(
        rewriter, iterationDirections, indicesAttr->getValue(),
        inductions);

    if (mlir::failed(cloneEquationTemplateIntoFunction(
            rewriter, symbolTableCollection, modelOp, equation.getTemplate(),
            shiftedInductions))) {
      return nullptr;
    }
  } else {
    if (mlir::failed(cloneEquationTemplateIntoFunction(
            rewriter, symbolTableCollection, modelOp, equation.getTemplate(),
            std::nullopt))) {
      return nullptr;
    }
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<YieldOp>(eqFunc.getLoc());
  return eqFunc;
}

llvm::SmallVector<mlir::Value> EquationExplicitationPass::shiftInductions(
    mlir::RewriterBase& rewriter,
    mlir::ArrayAttr iterationDirections,
    const MultidimensionalRange& indices,
    mlir::ValueRange loopInductions)
{
  llvm::SmallVector<mlir::Value> mappedInductions;

  for (size_t i = 0, e = indices.rank(); i < e; ++i) {
    mlir::Value induction = loopInductions[i];

    mlir::Value fromValue = rewriter.create<mlir::arith::ConstantOp>(
        induction.getLoc(),
        rewriter.getIndexAttr(indices[i].getBegin()));

    auto iterationDirectionAttr =
        iterationDirections[i].cast<EquationScheduleDirectionAttr>();

    auto iterationDirection = iterationDirectionAttr.getValue();

    if (iterationDirection == EquationScheduleDirection::Any ||
        iterationDirection == EquationScheduleDirection::Forward) {
      mlir::Value mappedInduction = rewriter.create<mlir::arith::AddIOp>(
          induction.getLoc(), rewriter.getIndexType(),
          fromValue, induction);

      mappedInductions.push_back(mappedInduction);
    } else {
      assert(iterationDirection == EquationScheduleDirection::Backward);
      mlir::Value mappedInduction = rewriter.create<mlir::arith::SubIOp>(
          induction.getLoc(), rewriter.getIndexType(),
          fromValue, induction);

      mappedInductions.push_back(mappedInduction);
    }
  }

  return mappedInductions;
}

mlir::LogicalResult
EquationExplicitationPass::cloneEquationTemplateIntoFunction(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    EquationTemplateOp templateOp,
    llvm::ArrayRef<mlir::Value> inductions)
{
  assert(templateOp.getInductionVariables().size() == inductions.size());
  mlir::IRMapping mapping;

  for (const auto& [oldInduction, newInduction] : llvm::zip(
           templateOp.getInductionVariables(), inductions)) {
    mapping.map(oldInduction, newInduction);
  }

  for (auto& op : templateOp.getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    }

    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      mlir::Value lhs = equationSidesOp.getLhsValues()[0];
      mlir::Value rhs = equationSidesOp.getRhsValues()[0];

      mlir::Value mappedLhs = mapping.lookup(lhs);
      mlir::Value mappedRhs = mapping.lookup(rhs);

      if (auto loadOp = mappedLhs.getDefiningOp<LoadOp>()) {
        // Left-hand side is a scalar element extracted from an array
        // variable.
        if (auto lhsType = mappedLhs.getType();
            lhsType != mappedRhs.getType()) {
          mappedRhs = rewriter.create<CastOp>(
              mappedRhs.getLoc(), mappedLhs.getType(), mappedRhs);
        }

        rewriter.create<StoreOp>(
            equationSidesOp.getLoc(), mappedRhs,
            loadOp.getArray(), loadOp.getIndices());

        continue;
      }

      if (auto getOp = lhs.getDefiningOp<VariableGetOp>()) {
        if (getOp.getType().isa<ArrayType>()) {
          rewriter.create<ArrayCopyOp>(
              equationSidesOp.getLoc(), mappedRhs, mapping.lookup(getOp));
        } else {
          // Left-hand side is a scalar variable.
          if (auto lhsType = mappedLhs.getType();
              lhsType != mappedRhs.getType()) {
            mappedRhs = rewriter.create<CastOp>(
                mappedRhs.getLoc(), mappedLhs.getType(), mappedRhs);
          }

          auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, getOp.getVariableAttr());

          rewriter.create<QualifiedVariableSetOp>(
              equationSidesOp.getLoc(), variableOp, mappedRhs);

          return mlir::success();
        }
      }
    } else if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variableGetOp.getVariableAttr());

      auto qualifiedVariableGetOp = rewriter.create<QualifiedVariableGetOp>(
          op.getLoc(), variableOp);

      mapping.map(variableGetOp.getResult(),
                  qualifiedVariableGetOp.getResult());
    } else {
      rewriter.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::getAccessAttrs(
    llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
    llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
    mlir::SymbolTableCollection& symbolTableCollection,
    ScheduledEquationInstanceOp equationOp)
{
  auto accessAnalysis =
      getVariableAccessAnalysis(equationOp, symbolTableCollection);

  if (!accessAnalysis) {
    return mlir::failure();
  }

  IndexSet equationIndices = equationOp.getIterationSpace();
  auto matchedAccess = equationOp.getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  IndexSet matchedVariableIndices =
      matchedAccess->getAccessFunction().map(equationIndices);

  auto writtenVariableAttr = VariableAttr::get(
      equationOp.getContext(),
      matchedAccess->getVariable(),
      IndexSetAttr::get(
          equationOp.getContext(),
          std::move(matchedVariableIndices)));

  writtenVariables.push_back(writtenVariableAttr);
  llvm::SmallVector<VariableAccess> readAccesses;

  auto accesses = accessAnalysis->get().getAccesses(
      equationOp, symbolTableCollection);

  if (!accesses) {
    return mlir::failure();
  }

  if (mlir::failed(equationOp.getReadAccesses(
          readAccesses, symbolTableCollection, *accesses))) {
    return mlir::failure();
  }

  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (const VariableAccess& readAccess : readAccesses) {
    const AccessFunction& accessFunction = readAccess.getAccessFunction();
    IndexSet readIndices = accessFunction.map(equationIndices);
    readVariablesIndices[readAccess.getVariable()] += readIndices;
  }

  for (const auto& entry : readVariablesIndices) {
    readVariables.push_back(VariableAttr::get(
        equationOp.getContext(),
        entry.getFirst(),
        IndexSetAttr::get(equationOp.getContext(), entry.getSecond())));
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationExplicitationPass()
  {
    return std::make_unique<EquationExplicitationPass>();
  }
}
