#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Scheduling.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SCHEDULINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::bridge;

namespace
{
  class SchedulingPass
      : public mlir::modelica::impl::SchedulingPassBase<SchedulingPass>
  {
    public:
      using SchedulingPassBase<SchedulingPass>::SchedulingPassBase;

      void runOnOperation() override;

    private:
      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          MatchedEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult processScheduleOp(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          ScheduleOp scheduleOp);

      mlir::LogicalResult processInitialModel(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          ScheduleOp scheduleOp,
          llvm::ArrayRef<InitialModelOp> initialModelOps);

      mlir::LogicalResult processMainModel(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          ScheduleOp scheduleOp,
          llvm::ArrayRef<MainModelOp> mainModelOps);

      mlir::LogicalResult processSCCs(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          ScheduleOp scheduleOp,
          llvm::ArrayRef<SCCOp> SCCs,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createContainerFn);
  };
}

void SchedulingPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    if (mlir::failed(processScheduleOp(
            symbolTableCollection, modelOp, scheduleOp))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult SchedulingPass::processScheduleOp(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    ScheduleOp scheduleOp)
{
  llvm::SmallVector<InitialModelOp> initialModelOps;
  llvm::SmallVector<MainModelOp> mainModelOps;

  for (auto& op : scheduleOp.getOps()) {
    if (auto initialModelOp = mlir::dyn_cast<InitialModelOp>(op)) {
      initialModelOps.push_back(initialModelOp);
      continue;
    }

    if (auto mainModelOp = mlir::dyn_cast<MainModelOp>(op)) {
      mainModelOps.push_back(mainModelOp);
      continue;
    }
  }

  if (mlir::failed(processInitialModel(
          symbolTableCollection, modelOp, scheduleOp, initialModelOps))) {
    return mlir::failure();
  }

  if (mlir::failed(processMainModel(
          symbolTableCollection, modelOp, scheduleOp, mainModelOps))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processInitialModel(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    ScheduleOp scheduleOp,
    llvm::ArrayRef<InitialModelOp> initialModelOps)
{
  // Collect the SCCs.
  llvm::SmallVector<SCCOp> SCCs;

  for (InitialModelOp initialModelOp : initialModelOps) {
    initialModelOp.collectSCCs(SCCs);
  }

  if (SCCs.empty()) {
    return mlir::success();
  }

  auto createContainerFn =
      [](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto initialModelOp = builder.create<InitialModelOp>(loc);
    builder.createBlock(&initialModelOp.getBodyRegion());
    return initialModelOp.getBody();
  };

  if (mlir::failed(processSCCs(
          symbolTableCollection, modelOp, scheduleOp, SCCs,
          createContainerFn))) {
    return mlir::failure();
  }

  // Erase the old equations containers.
  for (InitialModelOp initialModelOp : initialModelOps) {
    initialModelOp.erase();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processMainModel(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    ScheduleOp scheduleOp,
    llvm::ArrayRef<MainModelOp> mainModelOps)
{
  // Collect the SCCs.
  llvm::SmallVector<SCCOp> SCCs;

  for (MainModelOp mainModelOp : mainModelOps) {
    mainModelOp.collectSCCs(SCCs);
  }

  if (SCCs.empty()) {
    return mlir::success();
  }

  auto createContainerFn =
      [](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto mainModelOp = builder.create<MainModelOp>(loc);
    builder.createBlock(&mainModelOp.getBodyRegion());
    return mainModelOp.getBody();
  };

  if (mlir::failed(processSCCs(
          symbolTableCollection, modelOp, scheduleOp, SCCs,
          createContainerFn))) {
    return mlir::failure();
  }

  // Erase the old equations containers.
  for (MainModelOp mainModelOp : mainModelOps) {
    mainModelOp.erase();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processSCCs(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    ScheduleOp scheduleOp,
    llvm::ArrayRef<SCCOp> SCCs,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createContainerFn)
{
  // Compute the writes map.
  WritesMap<VariableOp, MatchedEquationInstanceOp> writesMap;

  if (mlir::failed(getWritesMap(
          writesMap, modelOp, SCCs, symbolTableCollection))) {
    return mlir::failure();
  }

  // Create the scheduler. We use the pointers to the real nodes in order to
  // speed up the copies.
  using Scheduler =
      ::marco::modeling::Scheduler<VariableBridge*, SCCBridge*>;

  Scheduler scheduler(&getContext());

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<std::unique_ptr<SCCBridge>> sccBridges;
  llvm::SmallVector<SCCBridge*> sccBridgePtrs;

  llvm::DenseMap<
      MatchedEquationInstanceOp, MatchedEquationBridge*> equationsMap;

  // Collect the variables.
  for (VariableOp variable : modelOp.getOps<VariableOp>()) {
    auto& bridge = variableBridges.emplace_back(
        VariableBridge::build(variable));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variable.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  // Collect the SCCs and the equations.
  for (SCCOp scc : SCCs) {
    llvm::SmallVector<MatchedEquationInstanceOp> equations;
    scc.collectEquations(equations);

    if (equations.empty()) {
      continue;
    }

    auto& sccBridge = sccBridges.emplace_back(SCCBridge::build(
        scc, symbolTableCollection, writesMap, equationsMap));

    sccBridgePtrs.push_back(sccBridge.get());

    for (MatchedEquationInstanceOp equation : equations) {
      auto variableAccessAnalysis =
          getVariableAccessAnalysis(equation, symbolTableCollection);

      if (!variableAccessAnalysis) {
        return mlir::failure();
      }

      auto& equationBridge = equationBridges.emplace_back(
          MatchedEquationBridge::build(
              equation, symbolTableCollection, *variableAccessAnalysis,
              variablesMap));

      equationsMap[equation] = equationBridge.get();
    }
  }

  // Compute the schedule.
  auto scheduledSCCs = scheduler.schedule(sccBridgePtrs);

  // Create the scheduled equations.
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(scheduleOp.getBody());

  mlir::Block* containerBody = createContainerFn(builder, modelOp.getLoc());
  builder.setInsertionPointToStart(containerBody);

  for (const auto& scc : scheduledSCCs) {
    bool hasCycle = scc.size() > 1;

    if (hasCycle) {
      // If the SCC contains a cycle, then all the equations must be declared
      // inside it.
      builder.setInsertionPointToEnd(containerBody);
      auto sccOp = builder.create<SCCOp>(scheduleOp.getLoc());
      mlir::Block* sccBody = builder.createBlock(&sccOp.getBodyRegion());
      builder.setInsertionPointToStart(sccBody);
    }

    for (const auto& scheduledEquation : scc) {
      MatchedEquationInstanceOp matchedEquation =
          scheduledEquation.getEquation()->op;

      size_t numOfInductions =
          matchedEquation.getInductionVariables().size();

      bool isScalarEquation = numOfInductions == 0;

      // Determine the iteration directions.
      llvm::SmallVector<mlir::Attribute, 3> iterationDirections;

      for (marco::modeling::scheduling::Direction direction :
           scheduledEquation.getIterationDirections()) {
        iterationDirections.push_back(
            EquationScheduleDirectionAttr::get(&getContext(), direction));
      }

      // Create an equation for each range of scheduled indices.
      const IndexSet& scheduledIndices = scheduledEquation.getIndexes();

      for (const MultidimensionalRange& scheduledRange :
           llvm::make_range(scheduledIndices.rangesBegin(),
                            scheduledIndices.rangesEnd())) {
        if (!hasCycle) {
          // If the SCC doesn't have a cycle, then each equation has to be
          // declared in a dedicated SCC operation.
          builder.setInsertionPointToEnd(containerBody);
          auto sccOp = builder.create<SCCOp>(scheduleOp.getLoc());
          mlir::Block* sccBody = builder.createBlock(&sccOp.getBodyRegion());
          builder.setInsertionPointToStart(sccBody);
        }

        // Create the operation for the scheduled equation.
        auto scheduledEquationOp =
            builder.create<ScheduledEquationInstanceOp>(
                matchedEquation.getLoc(),
                matchedEquation.getTemplate(),
                matchedEquation.getPath(),
                builder.getArrayAttr(llvm::ArrayRef(iterationDirections)
                                         .take_front(numOfInductions)));

        if (!isScalarEquation) {
          MultidimensionalRange explicitRange =
              scheduledRange.takeFirstDimensions(numOfInductions);

          scheduledEquationOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), explicitRange));
        }
      }
    }
  }

  return mlir::success();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getVariableAccessAnalysis(
    MatchedEquationInstanceOp equation,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  if (auto analysis = getCachedChildAnalysis<VariableAccessAnalysis>(
          equation.getTemplate())) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<VariableAccessAnalysis>(
      equation.getTemplate());

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSchedulingPass()
  {
    return std::make_unique<SchedulingPass>();
  }
}
