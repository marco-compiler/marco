#include "marco/Codegen/Transforms/InitialConditionsSolving.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_INITIALCONDITIONSSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class InitialConditionsSolvingPass
      : public mlir::modelica::impl::InitialConditionsSolvingPassBase<
            InitialConditionsSolvingPass>
  {
    public:
      using InitialConditionsSolvingPassBase<InitialConditionsSolvingPass>
          ::InitialConditionsSolvingPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);
  };
}

void InitialConditionsSolvingPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(moduleOp, modelOp))) {
      return signalPassFailure();
    }
  }
}

static mlir::LogicalResult addStartEquationsToSchedule(
    mlir::OpBuilder& builder,
    ScheduleOp schedule,
    llvm::ArrayRef<StartOp> startOps)
{
  // TODO
  return mlir::success();
}

mlir::LogicalResult InitialConditionsSolvingPass::processModelOp(
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<InitialModelOp> initialModelOps;
  llvm::SmallVector<SCCOp> SCCs;

  for (InitialModelOp initialModelOp : modelOp.getOps<InitialModelOp>()) {
    initialModelOps.push_back(initialModelOp);
  }

  for (InitialModelOp initialModelOp : initialModelOps) {
    initialModelOp.collectSCCs(SCCs);
  }

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "solveICModel",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!SCCs.empty()) {
    rewriter.setInsertionPointToEnd(modelOp.getBody());

    // Create the schedule operation.
    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "ic");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto initialModelOp = rewriter.create<InitialModelOp>(modelOp.getLoc());
    rewriter.createBlock(&initialModelOp.getBodyRegion());
    rewriter.setInsertionPointToStart(initialModelOp.getBody());

    for (SCCOp scc : SCCs) {
      rewriter.clone(*scc.getOperation());
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);

  for (InitialModelOp initialModelOp : initialModelOps) {
    rewriter.eraseOp(initialModelOp);
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createInitialConditionsSolvingPass()
  {
    return std::make_unique<InitialConditionsSolvingPass>();
  }
}
