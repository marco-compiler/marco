#include "marco/Dialect/BaseModelica/Transforms/EquationExplicitation.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationExplicitationPass
    : public impl::EquationExplicitationPassBase<EquationExplicitationPass> {
public:
  using EquationExplicitationPassBase<
      EquationExplicitationPass>::EquationExplicitationPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationInstanceOp equation,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult
  processScheduleOp(mlir::RewriterBase &rewriter,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    mlir::ModuleOp moduleOp, ModelOp modelOp,
                    ScheduleOp scheduleOp);

  mlir::LogicalResult convertStartEquations(ScheduleOp scheduleOp);

  mlir::LogicalResult
  processSCCs(mlir::RewriterBase &rewriter,
              mlir::SymbolTableCollection &symbolTableCollection,
              mlir::ModuleOp moduleOp, ModelOp modelOp,
              llvm::ArrayRef<SCCOp> SCCs);

  mlir::LogicalResult
  processSCC(mlir::RewriterBase &rewriter,
             mlir::SymbolTableCollection &symbolTableCollection,
             mlir::ModuleOp moduleOp, ModelOp modelOp, SCCOp scc);

  mlir::LogicalResult
  createEquationBlocks(mlir::RewriterBase &rewriter,
                       mlir::SymbolTableCollection &symbolTableCollection,
                       mlir::ModuleOp moduleOp, ModelOp modelOp, SCCOp sccOp,
                       EquationInstanceOp equation);

  EquationFunctionOp
  createEquationFunction(mlir::RewriterBase &rewriter,
                         mlir::SymbolTableCollection &symbolTableCollection,
                         mlir::ModuleOp moduleOp, ModelOp modelOp,
                         EquationInstanceOp equation, int64_t rank);

  mlir::LogicalResult cloneEquationTemplateIntoFunction(
      mlir::RewriterBase &rewriter,
      mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
      EquationTemplateOp templateOp, llvm::ArrayRef<mlir::Value> inductions);

  mlir::LogicalResult
  getAccessAttrs(llvm::SmallVectorImpl<Variable> &writtenVariables,
                 llvm::SmallVectorImpl<Variable> &readVariables,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 EquationInstanceOp equationOp);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void EquationExplicitationPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    for (ScheduleOp scheduleOp :
         llvm::make_early_inc_range(modelOp.getOps<ScheduleOp>())) {
      if (mlir::failed(processScheduleOp(rewriter, symbolTableCollection,
                                         moduleOp, modelOp, scheduleOp))) {
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
    EquationTemplateOp equationTemplate,
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = equationTemplate->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *op : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(op);
  }

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equationTemplate)) {
    return *analysis;
  }

  auto &analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equationTemplate);

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationExplicitationPass::getVariableAccessAnalysis(
    EquationInstanceOp equation,
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getVariableAccessAnalysis(equation.getTemplate(),
                                   symbolTableCollection);
}

mlir::LogicalResult EquationExplicitationPass::processScheduleOp(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, ScheduleOp scheduleOp) {
  llvm::SmallVector<StartEquationInstanceOp> startEquations;
  llvm::SmallVector<SCCOp> SCCs;

  if (mlir::failed(convertStartEquations(scheduleOp))) {
    return mlir::failure();
  }

  for (auto &op : scheduleOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOp.collectEquations(startEquations);
      initialOp.collectSCCs(SCCs);
      continue;
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(op)) {
      dynamicOp.collectSCCs(SCCs);
      continue;
    }
  }

  if (mlir::failed(processSCCs(rewriter, symbolTableCollection, moduleOp,
                               modelOp, SCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

namespace {
class StartEquationPattern
    : public mlir::OpConversionPattern<StartEquationInstanceOp> {
public:
  StartEquationPattern(mlir::MLIRContext *context,
                       mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::OpConversionPattern<StartEquationInstanceOp>(context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LogicalResult
  matchAndRewrite(StartEquationInstanceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sccOp = rewriter.replaceOpWithNewOp<SCCOp>(op);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    auto instanceOp =
        rewriter.create<EquationInstanceOp>(op.getLoc(), op.getTemplate());

    instanceOp.getProperties().indices = op.getProperties().indices;

    std::optional<VariableAccess> access = instanceOp.getAccessAtPath(
        *symbolTableCollection, EquationPath(EquationPath::LEFT, 0));

    if (!access) {
      return mlir::failure();
    }

    instanceOp.getProperties().match.name = access->getVariable();

    instanceOp.getProperties().match.indices =
        access->getAccessFunction().map(instanceOp.getProperties().indices);

    instanceOp.getProperties().schedule.resize(
        instanceOp.getProperties().indices.rank(), Schedule::Any);

    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
};
} // namespace

mlir::LogicalResult
EquationExplicitationPass::convertStartEquations(ScheduleOp scheduleOp) {
  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<StartEquationInstanceOp>();
  target.addLegalOp<EquationInstanceOp>();

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  patterns.insert<StartEquationPattern>(&getContext(), symbolTableCollection);

  return mlir::applyPartialConversion(scheduleOp, target, std::move(patterns));
}

mlir::LogicalResult EquationExplicitationPass::processSCCs(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs) {
  for (SCCOp scc : SCCs) {
    if (mlir::failed(processSCC(rewriter, symbolTableCollection, moduleOp,
                                modelOp, scc))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCC(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, SCCOp scc) {
  llvm::SmallVector<EquationInstanceOp> equations;
  scc.collectEquations(equations);

  if (equations.size() > 1) {
    // Cycle among the equations.
    return mlir::success();
  }

  bool isSCCErasable = true;

  for (EquationInstanceOp equation : equations) {
    EquationInstanceOp explicitEquation =
        equation.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (explicitEquation) {
      if (mlir::failed(createEquationBlocks(rewriter, symbolTableCollection,
                                            moduleOp, modelOp, scc,
                                            explicitEquation))) {
        return mlir::failure();
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

mlir::LogicalResult EquationExplicitationPass::createEquationBlocks(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, SCCOp sccOp, EquationInstanceOp equationOp) {
  EquationFunctionOp eqFunc = createEquationFunction(
      rewriter, symbolTableCollection, moduleOp, modelOp, equationOp,
      equationOp.getProperties().indices.rank());

  if (!eqFunc) {
    return mlir::failure();
  }

  rewriter.setInsertionPoint(sccOp);

  VariablesList writtenVariables;
  VariablesList readVariables;

  if (mlir::failed(getAccessAttrs(writtenVariables, readVariables,
                                  symbolTableCollection, equationOp))) {
    return mlir::failure();
  }

  auto scheduleBlockOp = rewriter.create<ScheduleBlockOp>(
      modelOp.getLoc(), true, writtenVariables, readVariables);

  rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

  bool independentIndices = !equationOp.getProperties().schedule.empty();

  if (independentIndices) {
    independentIndices &= llvm::all_of(
        equationOp.getProperties().schedule, [](Schedule schedule) {
          return schedule == marco::modeling::scheduling::Direction::Any;
        });
  }

  rewriter.create<EquationCallOp>(eqFunc.getLoc(), eqFunc.getSymName(),
                                  equationOp.getProperties().indices,
                                  independentIndices);

  return mlir::success();
}

EquationFunctionOp EquationExplicitationPass::createEquationFunction(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, EquationInstanceOp equation, int64_t rank) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto eqFunc = rewriter.create<EquationFunctionOp>(
      equation.getLoc(), "equation", equation.getInductionVariables().size());

  symbolTableCollection.getSymbolTable(moduleOp).insert(eqFunc);

  mlir::Block *entryBlock = eqFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> shiftedInductions;

  if (rank > 0) {
    const auto &iterationDirections = equation.getProperties().schedule;

    if (static_cast<int64_t>(iterationDirections.size()) != rank) {
      equation.emitOpError() << "Incompatible schedule";
      return nullptr;
    }

    llvm::SmallVector<mlir::Value> lowerBounds;
    llvm::SmallVector<mlir::Value> upperBounds;

    for (int64_t dim = 0; dim < rank; ++dim) {
      lowerBounds.push_back(eqFunc.getLowerBound(dim));
      upperBounds.push_back(eqFunc.getUpperBound(dim));
    }

    llvm::SmallVector<mlir::Value> inductions;

    mlir::Value oneValue = rewriter.create<mlir::arith::ConstantOp>(
        equation.getLoc(), rewriter.getIndexAttr(1));

    for (int64_t i = 0; i < rank; ++i) {
      auto forOp = rewriter.create<mlir::scf::ForOp>(
          equation.getLoc(), eqFunc.getLowerBound(i), eqFunc.getUpperBound(i),
          oneValue);

      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    for (size_t i = 0, e = inductions.size(); i < e; ++i) {
      mlir::Value induction = inductions[i];

      if (iterationDirections[i] == Schedule::Any ||
          iterationDirections[i] == Schedule::Forward) {
        shiftedInductions.push_back(induction);
      } else {
        assert(iterationDirections[i] == Schedule::Backward);

        mlir::Value offset = rewriter.create<mlir::arith::SubIOp>(
            induction.getLoc(), rewriter.getIndexType(), induction,
            lowerBounds[i]);

        mlir::Value mappedInduction = rewriter.create<mlir::arith::SubIOp>(
            induction.getLoc(), rewriter.getIndexType(), upperBounds[i],
            offset);

        shiftedInductions.push_back(mappedInduction);
      }
    }
  }

  if (mlir::failed(cloneEquationTemplateIntoFunction(
          rewriter, symbolTableCollection, modelOp, equation.getTemplate(),
          shiftedInductions))) {
    return nullptr;
  }

  // Replace the VariableGetOps.
  eqFunc.walk([&](VariableGetOp getOp) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    rewriter.setInsertionPoint(getOp);
    rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(getOp, variableOp);
  });

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<YieldOp>(eqFunc.getLoc());
  return eqFunc;
}

mlir::LogicalResult
EquationExplicitationPass::cloneEquationTemplateIntoFunction(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    EquationTemplateOp templateOp, llvm::ArrayRef<mlir::Value> inductions) {
  assert(templateOp.getInductionVariables().size() == inductions.size());
  mlir::IRMapping mapping;

  for (const auto &[oldInduction, newInduction] :
       llvm::zip(templateOp.getInductionVariables(), inductions)) {
    mapping.map(oldInduction, newInduction);
  }

  for (auto &op : templateOp.getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    }

    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);

      llvm::SmallVector<llvm::SmallVector<mlir::Value, 3>, 10> lhsSubscripts;
      mlir::Operation *lhsOldOp = lhs.getDefiningOp();

      while (lhsOldOp && !mlir::isa<QualifiedVariableGetOp>(lhsOldOp)) {
        if (auto tensorExtractOp = mlir::dyn_cast<TensorExtractOp>(lhsOldOp)) {
          auto subscripts = tensorExtractOp.getIndices();
          lhsSubscripts.emplace_back(subscripts.begin(), subscripts.end());
          lhsOldOp = tensorExtractOp.getTensor().getDefiningOp();
          continue;
        }

        if (auto tensorViewOp = mlir::dyn_cast<TensorViewOp>(lhsOldOp)) {
          auto subscripts = tensorViewOp.getSubscriptions();
          lhsSubscripts.emplace_back(subscripts.begin(), subscripts.end());
          lhsOldOp = tensorViewOp.getSource().getDefiningOp();
          continue;
        }

        llvm_unreachable("Unknown operation");
      }

      auto lhsGetOp = mlir::cast<QualifiedVariableGetOp>(lhsOldOp);

      if (lhsSubscripts.empty()) {
        rewriter.create<QualifiedVariableSetOp>(
            lhsGetOp.getLoc(), lhsGetOp.getVariable(), std::nullopt, rhs);
      } else {
        auto lhsTensorType =
            mlir::cast<mlir::TensorType>(lhsGetOp.getResult().getType());

        lhs = rewriter.create<QualifiedVariableGetOp>(
            lhsGetOp.getLoc(),
            ArrayType::get(lhsTensorType.getShape(),
                           lhsTensorType.getElementType()),
            lhsGetOp.getVariable());

        for (size_t i = 0, e = lhsSubscripts.size(); i < e; ++i) {
          lhs = rewriter.create<SubscriptionOp>(lhs.getLoc(), lhs,
                                                lhsSubscripts[e - i - 1]);
        }

        if (auto rhsArrayType = mlir::dyn_cast<ArrayType>(rhs.getType())) {
          rewriter.create<ArrayCopyOp>(equationSidesOp.getLoc(), rhs, lhs);
        } else {
          mlir::Type lhsElementType =
              mlir::cast<ArrayType>(lhs.getType()).getElementType();

          if (rhs.getType() != lhsElementType) {
            rhs = rewriter.create<CastOp>(rhs.getLoc(), lhsElementType, rhs);
          }

          rewriter.create<StoreOp>(equationSidesOp.getLoc(), rhs, lhs,
                                   std::nullopt);
        }
      }
    } else if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variableGetOp.getVariableAttr());

      auto qualifiedVariableGetOp =
          rewriter.create<QualifiedVariableGetOp>(op.getLoc(), variableOp);

      mapping.map(variableGetOp.getResult(),
                  qualifiedVariableGetOp.getResult());
    } else {
      rewriter.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::getAccessAttrs(
    llvm::SmallVectorImpl<Variable> &writtenVariables,
    llvm::SmallVectorImpl<Variable> &readVariables,
    mlir::SymbolTableCollection &symbolTableCollection,
    EquationInstanceOp equationOp) {
  auto accessAnalysis =
      getVariableAccessAnalysis(equationOp, symbolTableCollection);

  if (!accessAnalysis) {
    return mlir::failure();
  }

  IndexSet equationIndices = equationOp.getIterationSpace();

  writtenVariables.emplace_back(equationOp.getProperties().match.name,
                                equationOp.getProperties().match.indices);

  llvm::SmallVector<VariableAccess> readAccesses;

  auto accesses = accessAnalysis->get().getAccesses(symbolTableCollection);

  if (!accesses) {
    return mlir::failure();
  }

  if (mlir::failed(equationOp.getReadAccesses(
          readAccesses, symbolTableCollection, *accesses))) {
    return mlir::failure();
  }

  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (const VariableAccess &readAccess : readAccesses) {
    const AccessFunction &accessFunction = readAccess.getAccessFunction();
    IndexSet readIndices = accessFunction.map(equationIndices);
    readVariablesIndices[readAccess.getVariable()] += readIndices;
  }

  for (const auto &entry : readVariablesIndices) {
    readVariables.emplace_back(entry.getFirst(), entry.getSecond());
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());

  mlir::GreedyRewriteConfig config;
  config.fold = true;

  return mlir::applyPatternsGreedily(modelOp, std::move(patterns), config);
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationExplicitationPass() {
  return std::make_unique<EquationExplicitationPass>();
}
} // namespace mlir::bmodelica
