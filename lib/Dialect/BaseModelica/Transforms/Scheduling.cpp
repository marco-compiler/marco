#define DEBUG_TYPE "scheduling"

#include "marco/Dialect/BaseModelica/Transforms/Scheduling.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Scheduling.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCHEDULINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
class SchedulingPass
    : public mlir::bmodelica::impl::SchedulingPassBase<SchedulingPass> {
public:
  using SchedulingPassBase<SchedulingPass>::SchedulingPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(StartEquationInstanceOp equation,
                            mlir::SymbolTableCollection &symbolTableCollection);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(MatchedEquationInstanceOp equation,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  processScheduleOp(mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, ScheduleOp scheduleOp);

  mlir::LogicalResult
  processInitialModel(mlir::SymbolTableCollection &symbolTableCollection,
                      ModelOp modelOp, ScheduleOp scheduleOp,
                      llvm::ArrayRef<InitialOp> initialOps);

  mlir::LogicalResult
  processMainModel(mlir::SymbolTableCollection &symbolTableCollection,
                   ModelOp modelOp, ScheduleOp scheduleOp,
                   llvm::ArrayRef<DynamicOp> dynamicOps);

  mlir::LogicalResult
  schedule(mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
           ScheduleOp scheduleOp, llvm::ArrayRef<SCCOp> SCCs,
           llvm::ArrayRef<StartEquationInstanceOp> startEquations,
           llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
               createContainerFn);

  mlir::LogicalResult
  addStartEquations(mlir::OpBuilder &builder,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, mlir::Block *containerBody,
                    llvm::ArrayRef<StartEquationInstanceOp> startEquations);
};
} // namespace

void SchedulingPass::runOnOperation() {
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

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getVariableAccessAnalysis(
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
SchedulingPass::getVariableAccessAnalysis(
    StartEquationInstanceOp equation,
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getVariableAccessAnalysis(equation.getTemplate(),
                                   symbolTableCollection);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getVariableAccessAnalysis(
    MatchedEquationInstanceOp equation,
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getVariableAccessAnalysis(equation.getTemplate(),
                                   symbolTableCollection);
}

mlir::LogicalResult SchedulingPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;

  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    if (mlir::failed(
            processScheduleOp(symbolTableCollection, modelOp, scheduleOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processScheduleOp(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    ScheduleOp scheduleOp) {
  llvm::SmallVector<InitialOp> initialOps;
  llvm::SmallVector<DynamicOp> dynamicOps;

  for (auto &op : scheduleOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOps.push_back(initialOp);
      continue;
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(op)) {
      dynamicOps.push_back(dynamicOp);
      continue;
    }
  }

  if (mlir::failed(processInitialModel(symbolTableCollection, modelOp,
                                       scheduleOp, initialOps))) {
    return mlir::failure();
  }

  if (mlir::failed(processMainModel(symbolTableCollection, modelOp, scheduleOp,
                                    dynamicOps))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processInitialModel(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    ScheduleOp scheduleOp, llvm::ArrayRef<InitialOp> initialOps) {
  // Collect the start equations and the SCCs.
  llvm::SmallVector<StartEquationInstanceOp> startEquations;
  llvm::SmallVector<SCCOp> SCCs;

  for (InitialOp initialOp : initialOps) {
    initialOp.collectSCCs(SCCs);

    for (StartEquationInstanceOp startEquation :
         initialOp.getOps<StartEquationInstanceOp>()) {
      startEquations.push_back(startEquation);
    }
  }

  if (SCCs.empty()) {
    return mlir::success();
  }

  auto createContainerFn = [](mlir::OpBuilder &builder,
                              mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto initialOp = builder.create<InitialOp>(loc);
    builder.createBlock(&initialOp.getBodyRegion());
    return initialOp.getBody();
  };

  if (mlir::failed(schedule(symbolTableCollection, modelOp, scheduleOp, SCCs,
                            startEquations, createContainerFn))) {
    return mlir::failure();
  }

  // Erase the old equations containers.
  for (InitialOp initialOp : initialOps) {
    initialOp.erase();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::processMainModel(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    ScheduleOp scheduleOp, llvm::ArrayRef<DynamicOp> dynamicOps) {
  // Collect the SCCs.
  llvm::SmallVector<SCCOp> SCCs;

  for (DynamicOp dynamicOp : dynamicOps) {
    dynamicOp.collectSCCs(SCCs);
  }

  if (SCCs.empty()) {
    return mlir::success();
  }

  auto createContainerFn = [](mlir::OpBuilder &builder,
                              mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto dynamicOp = builder.create<DynamicOp>(loc);
    builder.createBlock(&dynamicOp.getBodyRegion());
    return dynamicOp.getBody();
  };

  if (mlir::failed(schedule(symbolTableCollection, modelOp, scheduleOp, SCCs,
                            std::nullopt, createContainerFn))) {
    return mlir::failure();
  }

  // Erase the old equations containers.
  for (DynamicOp dynamicOp : dynamicOps) {
    dynamicOp.erase();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::schedule(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    ScheduleOp scheduleOp, llvm::ArrayRef<SCCOp> SCCs,
    llvm::ArrayRef<StartEquationInstanceOp> startEquations,
    llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
        createContainerFn) {
  LLVM_DEBUG(llvm::dbgs() << "Performing scheduling on \""
                          << scheduleOp.getSymName() << "\"\n");

  // Compute the writes maps.
  WritesMap<VariableOp, MatchedEquationInstanceOp> matchedEquationsWritesMap;
  WritesMap<VariableOp, StartEquationInstanceOp> startEquationsWritesMap;

  if (mlir::failed(getWritesMap(matchedEquationsWritesMap, modelOp, SCCs,
                                symbolTableCollection))) {
    return mlir::failure();
  }

  if (mlir::failed(getWritesMap(startEquationsWritesMap, modelOp,
                                startEquations, symbolTableCollection))) {
    return mlir::failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Matched equations writes map:\n"
                          << matchedEquationsWritesMap << "\n");

  LLVM_DEBUG(llvm::dbgs() << "Start equations writes map:\n"
                          << startEquationsWritesMap << "\n");

  // Create the scheduler. We use the pointers to the real nodes in order to
  // speed up the copies.
  using Scheduler = ::marco::modeling::Scheduler<VariableBridge *, SCCBridge *>;

  Scheduler scheduler(&getContext());

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<std::unique_ptr<SCCBridge>> sccBridges;
  llvm::SmallVector<SCCBridge *> sccBridgePtrs;

  llvm::DenseMap<MatchedEquationInstanceOp, MatchedEquationBridge *>
      equationsMap;

  // Collect the variables.
  for (VariableOp variable : modelOp.getOps<VariableOp>()) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variable));

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

    auto &sccBridge = sccBridges.emplace_back(
        SCCBridge::build(scc, symbolTableCollection, matchedEquationsWritesMap,
                         startEquationsWritesMap, equationsMap));

    sccBridgePtrs.push_back(sccBridge.get());

    for (MatchedEquationInstanceOp equation : equations) {
      auto variableAccessAnalysis =
          getVariableAccessAnalysis(equation, symbolTableCollection);

      if (!variableAccessAnalysis) {
        return mlir::failure();
      }

      auto &equationBridge =
          equationBridges.emplace_back(MatchedEquationBridge::build(
              static_cast<int64_t>(equationBridges.size()), equation,
              symbolTableCollection, *variableAccessAnalysis, variablesMap));

      equationsMap[equation] = equationBridge.get();
    }
  }

  // Compute the schedule.
  auto scheduledSCCs = scheduler.schedule(sccBridgePtrs);

  // Create the scheduled equations.
  LLVM_DEBUG(llvm::dbgs() << "Creating the scheduled equations\n");

  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(scheduleOp.getBody());

  mlir::Block *containerBody = createContainerFn(builder, modelOp.getLoc());
  builder.setInsertionPointToStart(containerBody);

  for (const auto &scc : scheduledSCCs) {
    bool hasCycle = scc.size() > 1;

    if (hasCycle) {
      // If the SCC contains a cycle, then all the equations must be declared
      // inside it.
      builder.setInsertionPointToEnd(containerBody);
      auto sccOp = builder.create<SCCOp>(scheduleOp.getLoc());
      mlir::Block *sccBody = builder.createBlock(&sccOp.getBodyRegion());
      builder.setInsertionPointToStart(sccBody);
    }

    for (const auto &scheduledEquation : scc) {
      MatchedEquationInstanceOp matchedEquation =
          scheduledEquation.getEquation()->op;

      size_t numOfInductions = matchedEquation.getInductionVariables().size();

      bool isScalarEquation = numOfInductions == 0;

      // Determine the iteration directions.
      llvm::SmallVector<mlir::Attribute, 3> iterationDirections;

      for (marco::modeling::scheduling::Direction direction :
           scheduledEquation.getIterationDirections()) {
        iterationDirections.push_back(
            EquationScheduleDirectionAttr::get(&getContext(), direction));
      }

      // Create an equation for each range of scheduled indices.
      IndexSet scheduledIndices =
          scheduledEquation.getIndexes().getCanonicalRepresentation();

      for (const MultidimensionalRange &scheduledRange : llvm::make_range(
               scheduledIndices.rangesBegin(), scheduledIndices.rangesEnd())) {
        if (!hasCycle) {
          // If the SCC doesn't have a cycle, then each equation has to be
          // declared in a dedicated SCC operation.
          builder.setInsertionPointToEnd(containerBody);
          auto sccOp = builder.create<SCCOp>(scheduleOp.getLoc());
          mlir::Block *sccBody = builder.createBlock(&sccOp.getBodyRegion());
          builder.setInsertionPointToStart(sccBody);
        }

        // Create the operation for the scheduled equation.
        auto scheduledEquationOp = builder.create<ScheduledEquationInstanceOp>(
            matchedEquation.getLoc(), matchedEquation.getTemplate(),
            builder.getArrayAttr(llvm::ArrayRef(iterationDirections)
                                     .take_front(numOfInductions)));

        scheduledEquationOp.getProperties().match =
            matchedEquation.getProperties().match;

        if (!isScalarEquation) {
          MultidimensionalRange explicitRange =
              scheduledRange.takeFirstDimensions(numOfInductions);

          scheduledEquationOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), explicitRange));
        }
      }
    }
  }

  if (mlir::failed(addStartEquations(builder, symbolTableCollection, modelOp,
                                     containerBody, startEquations))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SchedulingPass::addStartEquations(
    mlir::OpBuilder &builder,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    mlir::Block *containerBody,
    llvm::ArrayRef<StartEquationInstanceOp> startEquations) {
  if (startEquations.empty()) {
    return mlir::success();
  }

  mlir::OpBuilder::InsertionGuard guard(builder);

  // Get the writes map.
  llvm::SmallVector<SCCOp> SCCs;

  for (SCCOp scc : containerBody->getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }

  WritesMap<VariableOp, SCCOp> writesMap;

  if (mlir::failed(getWritesMap<ScheduledEquationInstanceOp>(
          writesMap, modelOp, SCCs, symbolTableCollection))) {
    return mlir::failure();
  }

  // Determine the first SCC writing to each variable.
  llvm::DenseMap<VariableOp, SCCOp> firstWritingSCCs;

  for (const auto &entry : writesMap) {
    VariableOp writtenVariable = entry.first;
    SCCOp scc = entry.second.second;

    if (auto firstWritingSCCIt = firstWritingSCCs.find(writtenVariable);
        firstWritingSCCIt != firstWritingSCCs.end()) {
      if (scc->isBeforeInBlock(firstWritingSCCIt->getSecond())) {
        firstWritingSCCs[writtenVariable] = scc;
      }
    } else {
      firstWritingSCCs[writtenVariable] = scc;
    }
  }

  // Determine the last SCC computing the dependencies of each start equation.
  llvm::DenseMap<StartEquationInstanceOp, SCCOp> lastSCCDependencies;

  for (StartEquationInstanceOp equation : startEquations) {
    auto accessAnalysis =
        getVariableAccessAnalysis(equation, symbolTableCollection);

    if (!accessAnalysis) {
      return mlir::failure();
    }

    auto accesses =
        accessAnalysis->get().getAccesses(equation, symbolTableCollection);

    if (!accesses) {
      return mlir::failure();
    }

    IndexSet equationIndices = equation.getIterationSpace();
    llvm::SmallVector<VariableAccess> readAccesses;

    if (mlir::failed(equation.getReadAccesses(
            readAccesses, symbolTableCollection, *accesses))) {
      return mlir::failure();
    }

    for (const VariableAccess &readAccess : readAccesses) {
      auto readVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, readAccess.getVariable());

      if (!readVariableOp) {
        return mlir::failure();
      }

      const AccessFunction &accessFunction = readAccess.getAccessFunction();
      auto readIndices = accessFunction.map(equationIndices);

      for (const auto &writeEntry :
           llvm::make_range(writesMap.equal_range(readVariableOp))) {
        if (readIndices.empty() ||
            writeEntry.second.first.overlaps(readIndices)) {
          SCCOp writingSCC = writeEntry.second.second;

          if (auto lastSCCDependenciesIt = lastSCCDependencies.find(equation);
              lastSCCDependenciesIt != lastSCCDependencies.end()) {
            if (lastSCCDependenciesIt->second->isBeforeInBlock(writingSCC)) {
              lastSCCDependencies[equation] = writingSCC;
            }
          } else {
            lastSCCDependencies[equation] = writingSCC;
          }
        }
      }
    }
  }

  // Insert the start equations.
  for (StartEquationInstanceOp startEquation : startEquations) {
    auto writeAccess = startEquation.getWriteAccess(symbolTableCollection);

    if (!writeAccess) {
      return mlir::failure();
    }

    auto writtenVariable = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, writeAccess->getVariable());

    if (!writtenVariable) {
      return mlir::failure();
    }

    // Determine where to insert the start equation.
    auto lastDependencyIt = lastSCCDependencies.find(startEquation);

    if (lastDependencyIt == lastSCCDependencies.end()) {
      builder.setInsertionPointToStart(containerBody);
    } else {
      SCCOp lastDependency = lastDependencyIt->getSecond();
      builder.setInsertionPointAfter(lastDependency);

      auto firstWritingSCCIt = firstWritingSCCs.find(writtenVariable);

      if (firstWritingSCCIt != firstWritingSCCs.end()) {
        SCCOp firstWritingSCC = firstWritingSCCIt->getSecond();

        if (!lastDependency->isBeforeInBlock(firstWritingSCC)) {
          startEquation->emitWarning()
              << "the 'start' attribute will not be computed properly due to"
                 " cycles involving the its dependencies";
        }
      }
    }

    // Clone the equation instance.
    builder.clone(*startEquation.getOperation());
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSchedulingPass() {
  return std::make_unique<SchedulingPass>();
}
} // namespace mlir::bmodelica
