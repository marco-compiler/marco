#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Scheduling.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scheduling"

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
      using SchedulingPassBase::SchedulingPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processScheduleOp(ScheduleOp scheduleOp);

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          MatchedEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);
  };
}

void SchedulingPass::runOnOperation()
{
  ScheduleOp scheduleOp = getOperation();

  if (mlir::failed(processScheduleOp(scheduleOp))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult SchedulingPass::processScheduleOp(ScheduleOp scheduleOp)
{
  mlir::SymbolTableCollection symbolTableCollection;

  // Collect the SCCs.
  llvm::SmallVector<SCCOp> SCCs;
  scheduleOp.collectSCCs(SCCs);

  // Compute the writes map.
  auto moduleOp = scheduleOp->getParentOfType<mlir::ModuleOp>();
  WritesMap<SimulationVariableOp, MatchedEquationInstanceOp> writesMap;

  if (mlir::failed(getWritesMap(
          writesMap, moduleOp, scheduleOp, symbolTableCollection))) {
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
  for (SimulationVariableOp variable :
       moduleOp.getOps<SimulationVariableOp>()) {
    auto& bridge = variableBridges.emplace_back(
        VariableBridge::build(variable));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variable.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  // Collect the SCCs and the equations.
  for (SCCOp scc : scheduleOp.getOps<SCCOp>()) {
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
  auto sccGroups = scheduler.schedule(sccBridgePtrs);

  // Write the schedule and erase the old SCCs.
  mlir::IRRewriter rewriter(&getContext());
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  for (const auto& sccGroup : sccGroups) {
    rewriter.setInsertionPointToEnd(scheduleOp.getBody());

    // Create the group of independent SCCs.
    auto sccGroupOp = rewriter.create<SCCGroupOp>(scheduleOp.getLoc());
    assert(sccGroupOp.getBodyRegion().empty());

    mlir::Block* sccGroupBody =
        rewriter.createBlock(&sccGroupOp.getBodyRegion());

    for (const auto& scc : sccGroup) {
      bool hasCycle = scc.size() > 1;

      if (hasCycle) {
        // If the SCC contains a cycle, then all the equations must be declared
        // inside it.
        rewriter.setInsertionPointToEnd(sccGroupBody);
        auto sccOp = rewriter.create<SCCOp>(scheduleOp.getLoc());
        mlir::Block* sccBody = rewriter.createBlock(&sccOp.getBodyRegion());
        rewriter.setInsertionPointToStart(sccBody);
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
            rewriter.setInsertionPointToEnd(sccGroupBody);
            auto sccOp = rewriter.create<SCCOp>(scheduleOp.getLoc());
            mlir::Block* sccBody = rewriter.createBlock(&sccOp.getBodyRegion());
            rewriter.setInsertionPointToStart(sccBody);
          }

          // Create the operation for the scheduled equation.
          auto scheduledEquationOp =
              rewriter.create<ScheduledEquationInstanceOp>(
                  matchedEquation.getLoc(),
                  matchedEquation.getTemplate(),
                  matchedEquation.getPath(),
                  rewriter.getArrayAttr(llvm::ArrayRef(iterationDirections)
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
  }

  // Erase the old SCCs.
  for (SCCOp scc : SCCs) {
    rewriter.eraseOp(scc);
  }

  return mlir::success();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getVariableAccessAnalysis(
    MatchedEquationInstanceOp equation,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  llvm::SmallVector<mlir::Operation*> parents;
  mlir::Operation* parent = equation.getTemplate()->getParentOp();

  while (parent != nullptr && parent != getOperation()) {
    parents.push_back(parent);
    parent = parent->getParentOp();
  }

  auto analysisManager = getAnalysisManager();

  while (!parents.empty()) {
    analysisManager = analysisManager.nest(parents.pop_back_val());
  }

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

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSchedulingPass()
  {
    return std::make_unique<SchedulingPass>();
  }
}
