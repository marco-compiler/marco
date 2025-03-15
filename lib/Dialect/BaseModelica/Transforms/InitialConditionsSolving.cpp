#include "marco/Dialect/BaseModelica/Transforms/InitialConditionsSolving.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_INITIALCONDITIONSSOLVINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class InitialConditionsSolvingPass
    : public mlir::bmodelica::impl::InitialConditionsSolvingPassBase<
          InitialConditionsSolvingPass> {
public:
  using InitialConditionsSolvingPassBase<
      InitialConditionsSolvingPass>::InitialConditionsSolvingPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult
  processModelOp(mlir::SymbolTableCollection &symbolTableCollection,
                 mlir::ModuleOp moduleOp, ModelOp modelOp);
};
} // namespace

void InitialConditionsSolvingPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(
            processModelOp(symbolTableCollection, moduleOp, modelOp))) {
      return signalPassFailure();
    }
  }
}

static EquationTemplateOp
createEquationTemplate(mlir::RewriterBase &rewriter,
                       mlir::SymbolTableCollection &symbolTableCollection,
                       ModelOp modelOp, StartOp startOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(modelOp.getBody());

  auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, startOp.getVariableAttr());

  if (!variableOp) {
    return nullptr;
  }

  VariableType variableType = variableOp.getVariableType();
  int64_t numOfInductions = variableType.getRank();

  auto templateOp = rewriter.create<EquationTemplateOp>(startOp.getLoc());
  mlir::Block *templateBody = templateOp.createBody(numOfInductions);
  rewriter.mergeBlocks(startOp.getBody(), templateBody);

  auto clonedYieldedOp = mlir::cast<YieldOp>(templateBody->getTerminator());
  mlir::Value clonedYieldedValue = clonedYieldedOp.getValues()[0];
  rewriter.setInsertionPoint(clonedYieldedOp);

  mlir::Value lhs =
      rewriter.create<VariableGetOp>(startOp.getLoc(), variableOp);

  mlir::Value rhs = clonedYieldedValue;

  if (!variableType.isScalar()) {
    lhs = rewriter.create<TensorExtractOp>(lhs.getLoc(), lhs,
                                           templateOp.getInductionVariables());
  }

  if (auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType())) {
    int64_t rhsRank = rhsTensorType.getRank();

    rhs = rewriter.create<TensorExtractOp>(
        rhs.getLoc(), rhs,
        templateOp.getInductionVariables().take_back(rhsRank));
  }

  mlir::Value lhsOp = rewriter.create<EquationSideOp>(lhs.getLoc(), lhs);
  mlir::Value rhsOp = rewriter.create<EquationSideOp>(rhs.getLoc(), rhs);
  rewriter.replaceOpWithNewOp<EquationSidesOp>(clonedYieldedOp, lhsOp, rhsOp);

  return templateOp;
}

static mlir::LogicalResult
addStartEquationsToSchedule(mlir::RewriterBase &rewriter,
                            mlir::SymbolTableCollection &symbolTableCollection,
                            ModelOp modelOp, InitialOp initialOp,
                            llvm::ArrayRef<StartOp> startOps) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  for (StartOp startOp : startOps) {
    assert(!startOp.getFixed());

    auto templateOp = createEquationTemplate(rewriter, symbolTableCollection,
                                             modelOp, startOp);

    if (!templateOp) {
      return mlir::failure();
    }

    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, startOp.getVariableAttr());

    if (!variableOp) {
      return mlir::failure();
    }

    IndexSet variableIndices =
        variableOp.getIndices().getCanonicalRepresentation();

    rewriter.setInsertionPointToStart(initialOp.getBody());

    auto instanceOp =
        rewriter.create<StartEquationInstanceOp>(startOp.getLoc(), templateOp);

    instanceOp.getProperties().setIndices(variableIndices);
  }

  return mlir::success();
}

mlir::LogicalResult InitialConditionsSolvingPass::processModelOp(
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<StartOp> unfixedStartOps;
  llvm::SmallVector<InitialOp> initialOps;
  llvm::SmallVector<SCCOp> SCCs;

  for (auto &op : modelOp.getOps()) {
    if (auto startOp = mlir::dyn_cast<StartOp>(op)) {
      if (!startOp.getFixed() && !startOp.getImplicit()) {
        unfixedStartOps.push_back(startOp);
      }

      continue;
    }

    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOps.push_back(initialOp);
      continue;
    }
  }

  for (InitialOp initialOp : initialOps) {
    initialOp.collectSCCs(SCCs);
  }

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "solveICModel",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!SCCs.empty() || !unfixedStartOps.empty()) {
    rewriter.setInsertionPointToEnd(modelOp.getBody());

    // Create the schedule operation.
    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "ic");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto initialOp = rewriter.create<InitialOp>(modelOp.getLoc());
    rewriter.createBlock(&initialOp.getBodyRegion());
    rewriter.setInsertionPointToStart(initialOp.getBody());

    if (mlir::failed(addStartEquationsToSchedule(rewriter,
                                                 symbolTableCollection, modelOp,
                                                 initialOp, unfixedStartOps))) {
      return mlir::failure();
    }

    for (SCCOp scc : SCCs) {
      rewriter.clone(*scc.getOperation());
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);

  for (InitialOp initialOp : initialOps) {
    rewriter.eraseOp(initialOp);
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createInitialConditionsSolvingPass() {
  return std::make_unique<InitialConditionsSolvingPass>();
}
} // namespace mlir::bmodelica
