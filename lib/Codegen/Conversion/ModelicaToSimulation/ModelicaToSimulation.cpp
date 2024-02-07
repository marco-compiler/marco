#include "marco/Codegen/Conversion/ModelicaToSimulation/ModelicaToSimulation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOSIMULATIONCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ModelicaToSimulationConversionPass
      : public mlir::impl::ModelicaToSimulationConversionPassBase<
            ModelicaToSimulationConversionPass>
  {
    public:
      using ModelicaToSimulationConversionPassBase<
          ModelicaToSimulationConversionPass>
          ::ModelicaToSimulationConversionPassBase;

      void runOnOperation() override;

    private:
      DerivativesMap& getDerivativesMap(ModelOp modelOp);

      mlir::LogicalResult addMissingSimulationFunctions(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp);

      mlir::LogicalResult processModelOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult convertSchedules(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult convertScheduleBodyOp(
          mlir::RewriterBase& rewriter,
          mlir::IRMapping& mapping,
          mlir::Operation* op);

      mlir::LogicalResult convertScheduleBodyOp(
          mlir::RewriterBase& rewriter,
          mlir::IRMapping& mapping,
          InitialModelOp op);

      mlir::LogicalResult convertScheduleBodyOp(
          mlir::RewriterBase& rewriter,
          mlir::IRMapping& mapping,
          MainModelOp op);

      mlir::LogicalResult convertScheduleBodyOp(
          mlir::RewriterBase& rewriter,
          mlir::IRMapping& mapping,
          ParallelScheduleBlocksOp op);

      mlir::LogicalResult convertScheduleBodyOp(
          mlir::RewriterBase& rewriter,
          mlir::IRMapping& mapping,
          ScheduleBlockOp op);

      mlir::LogicalResult createModelNameOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult createNumOfVariablesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      mlir::LogicalResult createVariableNamesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      mlir::LogicalResult createVariableRanksOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      mlir::LogicalResult createPrintableIndicesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables,
          const marco::VariableFilter& variablesFilter);

      mlir::LogicalResult createDerivativesMapOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      mlir::LogicalResult createVariableGetters(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      /// Create the function that is called before starting the simulation.
      mlir::LogicalResult createInitFunction(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables);

      /// Create the function that is called when the simulation has finished.
      mlir::LogicalResult createDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult declareGlobalVariables(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<VariableOp> variables,
          llvm::StringMap<GlobalVariableOp>& globalVariablesMap);

      mlir::LogicalResult convertQualifiedVariableAccesses(
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::StringMap<GlobalVariableOp>& globalVariablesMap);

      GlobalVariableOp declareTimeVariable(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult createTimeGetterOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          GlobalVariableOp timeVariableOp);

      mlir::LogicalResult createTimeSetterOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          GlobalVariableOp timeVariableOp);

      mlir::LogicalResult convertTimeOp(mlir::ModuleOp moduleOp);
  };
}

void ModelicaToSimulationConversionPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  if (mlir::failed(addMissingSimulationFunctions(rewriter, moduleOp))) {
    return signalPassFailure();
  }

  llvm::SmallVector<ModelOp> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  if (modelOps.empty()) {
    moduleOp.emitError() << "no model found";
    return signalPassFailure();
  }

  if (modelOps.size() > 1) {
    moduleOp.emitError() << "more than one model found";
    return signalPassFailure();
  }

  if (mlir::failed(processModelOp(
          rewriter, symbolTableCollection, moduleOp, modelOps[0]))) {
    return signalPassFailure();
  }

  symbolTableCollection.getSymbolTable(moduleOp).remove(modelOps[0]);
  rewriter.eraseOp(modelOps[0]);

  // Declare the time variable.
  GlobalVariableOp timeVariableOp =
      declareTimeVariable(rewriter, moduleOp, symbolTableCollection);

  if (!timeVariableOp) {
    return signalPassFailure();
  }

  // Declare the time getter.
  if (mlir::failed(createTimeGetterOp(
          rewriter, moduleOp, symbolTableCollection, timeVariableOp))) {
    return signalPassFailure();
  }

  // Declare the time setter.
  if (mlir::failed(createTimeSetterOp(
          rewriter, moduleOp, symbolTableCollection, timeVariableOp))) {
    return signalPassFailure();
  }

  // Convert the time operation.
  if (mlir::failed(convertTimeOp(moduleOp))) {
    return signalPassFailure();
  }
}

DerivativesMap& ModelicaToSimulationConversionPass::getDerivativesMap(
    ModelOp modelOp)
{
  if (auto analysis = getCachedChildAnalysis<DerivativesMap>(modelOp)) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<DerivativesMap>(modelOp);
  analysis.initialize();
  return analysis;
}

mlir::LogicalResult
ModelicaToSimulationConversionPass::addMissingSimulationFunctions(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  size_t numOfICModelBeginOps = 0, numOfICModelEndOps = 0;
  size_t numOfDynamicModelBeginOps = 0, numOfDynamicModelEndOps = 0;

  for (auto& op : moduleOp.getOps()) {
    if (mlir::isa<mlir::simulation::ICModelBeginOp>(op)) {
      ++numOfICModelBeginOps;
    } else if (mlir::isa<mlir::simulation::ICModelEndOp>(op)) {
      ++numOfICModelEndOps;
    } else if (mlir::isa<mlir::simulation::DynamicModelBeginOp>(op)) {
      ++numOfDynamicModelBeginOps;
    } else if (mlir::isa<mlir::simulation::DynamicModelEndOp>(op)) {
      ++numOfDynamicModelEndOps;
    }
  }

  if (numOfICModelBeginOps == 0) {
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto op = builder.create<mlir::simulation::ICModelBeginOp>(
        moduleOp.getLoc());

    builder.createBlock(&op.getBodyRegion());
  }

  if (numOfICModelEndOps == 0) {
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto op = builder.create<mlir::simulation::ICModelEndOp>(
        moduleOp.getLoc());

    builder.createBlock(&op.getBodyRegion());
  }

  if (numOfDynamicModelBeginOps == 0) {
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto op = builder.create<mlir::simulation::DynamicModelBeginOp>(
        moduleOp.getLoc());

    builder.createBlock(&op.getBodyRegion());
  }

  if (numOfDynamicModelEndOps == 0) {
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto op = builder.create<mlir::simulation::DynamicModelEndOp>(
        moduleOp.getLoc());

    builder.createBlock(&op.getBodyRegion());
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::processModelOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  if (mlir::failed(convertSchedules(
          rewriter, symbolTableCollection, moduleOp, modelOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createModelNameOp(
          rewriter, moduleOp, modelOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createNumOfVariablesOp(
          rewriter, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  if (mlir::failed(createVariableNamesOp(
          rewriter, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  if (mlir::failed(createVariableRanksOp(
          rewriter, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  auto expectedVariablesFilter =
      marco::VariableFilter::fromString(variablesFilter);

  std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

  if (!expectedVariablesFilter) {
    getOperation().emitWarning(
        "Invalid variable filter string. No filtering will take place");

    variablesFilterInstance = std::make_unique<marco::VariableFilter>();
  } else {
    variablesFilterInstance = std::make_unique<marco::VariableFilter>(
        std::move(*expectedVariablesFilter));
  }

  if (mlir::failed(createPrintableIndicesOp(
          rewriter, moduleOp, modelOp, variables, *variablesFilterInstance))) {
    return mlir::failure();
  }

  if (mlir::failed(createDerivativesMapOp(
          rewriter, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  if (mlir::failed(createVariableGetters(
          rewriter, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitFunction(rewriter, moduleOp, modelOp))) {
    return mlir::failure();
  }

  llvm::StringMap<GlobalVariableOp> globalVariablesMap;

  if (mlir::failed(declareGlobalVariables(
          rewriter, symbolTableCollection, moduleOp, variables,
          globalVariablesMap))) {
    return mlir::failure();
  }

  if (mlir::failed(convertQualifiedVariableAccesses(
          symbolTableCollection, moduleOp, modelOp, globalVariablesMap))) {
    return mlir::failure();
  }

  return mlir::success();
}

static std::string flattenScheduleName(mlir::SymbolRefAttr name)
{
  std::string result = name.getRootReference().str();

  for (mlir::FlatSymbolRefAttr nestedRef : name.getNestedReferences()) {
    result += "_" + nestedRef.getValue().str();
  }

  return result;
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertSchedules(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  llvm::DenseMap<ScheduleOp, llvm::SmallVector<RunScheduleOp, 1>> schedules;

  moduleOp.walk([&](RunScheduleOp runScheduleOp) {
    ScheduleOp scheduleOp = runScheduleOp.getScheduleOp(symbolTableCollection);
    schedules[scheduleOp].push_back(runScheduleOp);
  });

  for (const auto& entry : schedules) {
    ScheduleOp scheduleOp = entry.getFirst();
    auto qualifiedName = getSymbolRefFromRoot(scheduleOp);

    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        scheduleOp.getLoc(), flattenScheduleName(qualifiedName),
        rewriter.getFunctionType(std::nullopt, std::nullopt));

    symbolTableCollection.getSymbolTable(moduleOp).insert(funcOp);
    mlir::Block* entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    mlir::IRMapping mapping;

    for (auto& nestedOp : scheduleOp.getOps()) {
      if (mlir::failed(convertScheduleBodyOp(rewriter, mapping, &nestedOp))) {
        return mlir::failure();
      }
    }

    rewriter.create<mlir::func::ReturnOp>(scheduleOp.getLoc());

    for (RunScheduleOp runScheduleOp : schedules[scheduleOp]) {
      rewriter.setInsertionPoint(runScheduleOp);

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          runScheduleOp, funcOp, std::nullopt);
    }

    symbolTableCollection.getSymbolTable(modelOp).remove(scheduleOp);
    rewriter.eraseOp(scheduleOp);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertScheduleBodyOp(
    mlir::RewriterBase& rewriter,
    mlir::IRMapping& mapping,
    InitialModelOp op)
{
  for (auto& nestedOp : op.getOps()) {
    if (auto parallelScheduleBlocksOp =
            mlir::dyn_cast<ParallelScheduleBlocksOp>(nestedOp)) {
      if (mlir::failed(convertScheduleBodyOp(
              rewriter, mapping, parallelScheduleBlocksOp))) {
        return mlir::failure();
      }

      continue;
    }

    if (auto scheduleBlockOp = mlir::dyn_cast<ScheduleBlockOp>(nestedOp)) {
      if (mlir::failed(convertScheduleBodyOp(
              rewriter, mapping, scheduleBlockOp))) {
        return mlir::failure();
      }

      continue;
    }

    rewriter.clone(nestedOp, mapping);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertScheduleBodyOp(
    mlir::RewriterBase& rewriter,
    mlir::IRMapping& mapping,
    MainModelOp op)
{
  for (auto& nestedOp : op.getOps()) {
    if (auto parallelScheduleBlocksOp =
            mlir::dyn_cast<ParallelScheduleBlocksOp>(nestedOp)) {
      if (mlir::failed(convertScheduleBodyOp(
              rewriter, mapping, parallelScheduleBlocksOp))) {
        return mlir::failure();
      }

      continue;
    }

    if (auto scheduleBlockOp = mlir::dyn_cast<ScheduleBlockOp>(nestedOp)) {
      if (mlir::failed(convertScheduleBodyOp(
              rewriter, mapping, scheduleBlockOp))) {
        return mlir::failure();
      }

      continue;
    }

    rewriter.clone(nestedOp, mapping);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertScheduleBodyOp(
    mlir::RewriterBase& rewriter,
    mlir::IRMapping& mapping,
    mlir::Operation* op)
{
  if (auto initialModelOp = mlir::dyn_cast<InitialModelOp>(op)) {
    return convertScheduleBodyOp(rewriter, mapping, initialModelOp);
  }

  if (auto mainModelOp = mlir::dyn_cast<MainModelOp>(op)) {
    return convertScheduleBodyOp(rewriter, mapping, mainModelOp);
  }

  rewriter.clone(*op, mapping);
  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertScheduleBodyOp(
    mlir::RewriterBase& rewriter,
    mlir::IRMapping& mapping,
    ParallelScheduleBlocksOp op)
{
  for (auto& nestedOp : op.getOps()) {
    if (auto scheduleBlockOp = mlir::dyn_cast<ScheduleBlockOp>(nestedOp)) {
      if (mlir::failed(convertScheduleBodyOp(
              rewriter, mapping, scheduleBlockOp))) {
        return mlir::failure();
      }

      continue;
    }

    rewriter.clone(nestedOp, mapping);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertScheduleBodyOp(
    mlir::RewriterBase& rewriter,
    mlir::IRMapping& mapping,
    ScheduleBlockOp op)
{
  for (auto& nestedOp : op.getOps()) {
    rewriter.clone(nestedOp, mapping);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createModelNameOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  builder.create<mlir::simulation::ModelNameOp>(
      modelOp.getLoc(), modelOp.getSymName());

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createNumOfVariablesOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto numOfVariables = static_cast<int64_t>(variables.size());

  builder.create<mlir::simulation::NumberOfVariablesOp>(
      modelOp.getLoc(), builder.getI64IntegerAttr(numOfVariables));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createVariableNamesOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<mlir::Attribute> names;

  for (VariableOp variable : variables) {
    names.push_back(builder.getStringAttr(variable.getSymName()));
  }

  builder.create<mlir::simulation::VariableNamesOp>(
      modelOp.getLoc(), builder.getArrayAttr(names));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createVariableRanksOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<int64_t> ranks;

  for (VariableOp variable : variables) {
    VariableType variableType = variable.getVariableType();
    ranks.push_back(variableType.getRank());
  }

  builder.create<mlir::simulation::VariableRanksOp>(
      modelOp.getLoc(), builder.getI64ArrayAttr(ranks));

  return mlir::success();
}

static IndexSet getPrintableIndices(
    VariableType variableType,
    llvm::ArrayRef<marco::VariableFilter::Filter> filters)
{
  assert(!variableType.isScalar());
  IndexSet result;

  for (const auto& filter : filters) {
    if (!filter.isVisible()) {
      continue;
    }

    auto filterRanges = filter.getRanges();
    llvm::SmallVector<Range, 3> ranges;

    assert(variableType.hasStaticShape());

    assert(static_cast<int64_t>(filterRanges.size()) ==
           variableType.getRank());

    for (const auto& range : llvm::enumerate(filterRanges)) {
      // In Modelica, arrays are 1-based. If present, we need to lower by 1 the
      // value given by the variable filter.

      auto lowerBound = range.value().hasLowerBound()
          ? range.value().getLowerBound() - 1 : 0;

      auto upperBound = range.value().hasUpperBound()
          ? range.value().getUpperBound()
          : variableType.getShape()[range.index()];

      ranges.emplace_back(lowerBound, upperBound);
    }

    result += MultidimensionalRange(std::move(ranges));
  }

  return std::move(result);
}

mlir::LogicalResult
ModelicaToSimulationConversionPass::createPrintableIndicesOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables,
    const marco::VariableFilter& variablesFilter)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<mlir::Attribute> printInformation;

  auto getFlatName = [](mlir::SymbolRefAttr symbolRef) -> std::string {
    std::string result = symbolRef.getRootReference().str();

    for (mlir::FlatSymbolRefAttr nested : symbolRef.getNestedReferences()) {
      result += "." + nested.getValue().str();
    }

    return result;
  };

  auto& derivativesMap = getDerivativesMap(modelOp);

  for (VariableOp variableOp : variables) {
    VariableType variableType = variableOp.getVariableType();
    std::vector<marco::VariableFilter::Filter> filters;

    if (auto stateName = derivativesMap.getDerivedVariable(
            mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      // Derivative variable.
      filters = variablesFilter.getVariableDerInfo(
          getFlatName(*stateName), variableType.getRank());

    } else {
      // Non-derivative variable.
      filters = variablesFilter.getVariableInfo(
          variableOp.getSymName(), variableType.getRank());
    }

    if (variableType.isScalar()) {
      // Scalar variable.
      bool isVisible = llvm::any_of(
          filters, [](const marco::VariableFilter::Filter& filter) {
            return filter.isVisible();
          });

      printInformation.push_back(builder.getBoolAttr(isVisible));
    } else {
      // Array variable.
      IndexSet printableIndices = getPrintableIndices(variableType, filters);
      printableIndices = printableIndices.getCanonicalRepresentation();

      printInformation.push_back(IndexSetAttr::get(
          builder.getContext(), printableIndices));
    }
  }

  builder.create<mlir::simulation::PrintableIndicesOp>(
      modelOp.getLoc(), builder.getArrayAttr(printInformation));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createDerivativesMapOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Map the position of the variables for faster lookups.
  llvm::DenseMap<mlir::SymbolRefAttr, size_t> positionsMap;

  for (size_t i = 0, e = variables.size(); i < e; ++i) {
    VariableOp variableOp = variables[i];
    auto name = mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr());
    positionsMap[name] = i;
  }

  // Compute the positions of the derivatives.
  llvm::SmallVector<int64_t> derivatives;
  auto& derivativesMap = getDerivativesMap(modelOp);

  for (VariableOp variable : variables) {
    if (auto derivative = derivativesMap.getDerivative(
            mlir::FlatSymbolRefAttr::get(variable.getSymNameAttr()))) {
      auto it = positionsMap.find(*derivative);

      if (it == positionsMap.end()) {
        return mlir::failure();
      }

      derivatives.push_back(static_cast<int64_t>(it->getSecond()));
    } else {
      derivatives.push_back(-1);
    }
  }

  builder.create<mlir::simulation::DerivativesMapOp>(
      modelOp.getLoc(), builder.getI64ArrayAttr(derivatives));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createVariableGetters(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Create a getter for each variable.
  size_t variableGetterCounter = 0;
  llvm::SmallVector<mlir::Attribute> getterNames;

  for (VariableOp variable : variables) {
    builder.setInsertionPointToEnd(moduleOp.getBody());
    VariableType variableType = variable.getVariableType();

    auto getterOp = builder.create<mlir::simulation::VariableGetterOp>(
        variable.getLoc(),
        "var_getter_" + std::to_string(variableGetterCounter++),
        variableType.getRank());

    getterNames.push_back(
        mlir::FlatSymbolRefAttr::get(getterOp.getSymNameAttr()));

    mlir::Block* bodyBlock = getterOp.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    mlir::Value getOp = builder.create<QualifiedVariableGetOp>(
        variable.getLoc(), variable);

    mlir::Value result = getOp;

    if (result.getType().isa<ArrayType>()) {
      result = builder.create<LoadOp>(
          result.getLoc(), result, getterOp.getIndices());
    }

    result = builder.create<CastOp>(
        result.getLoc(), getterOp.getResultTypes()[0], result);

    builder.create<mlir::simulation::ReturnOp>(result.getLoc(), result);
  }

  // Create the operation collecting all the getters.
  builder.setInsertionPointToEnd(moduleOp.getBody());

  builder.create<mlir::simulation::VariableGettersOp>(
      modelOp.getLoc(), builder.getArrayAttr(getterNames));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createInitFunction(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto initFunctionOp =
      rewriter.create<mlir::simulation::InitFunctionOp>(modelOp.getLoc());

  mlir::Block* entryBlock =
      rewriter.createBlock(&initFunctionOp.getBodyRegion());

  rewriter.setInsertionPointToStart(entryBlock);

  // Initialize the variables to zero.
  for (VariableOp variable : variables) {
    VariableType variableType = variable.getVariableType();

    auto constantMaterializableElementType =
        variableType.getElementType().dyn_cast<ConstantMaterializableType>();

    if (!constantMaterializableElementType) {
      return mlir::failure();
    }

    mlir::Value zeroValue =
        constantMaterializableElementType.materializeIntConstant(
            rewriter, variable.getLoc(), 0);

    if (variableType.isScalar()) {
      rewriter.create<QualifiedVariableSetOp>(
          variable.getLoc(), variable, zeroValue);
    } else {
      mlir::Value destination = rewriter.create<QualifiedVariableGetOp>(
          variable.getLoc(), variable);

      rewriter.create<ArrayFillOp>(destination.getLoc(), destination, zeroValue);
    }
  }

  rewriter.setInsertionPointToEnd(&initFunctionOp.getBodyRegion().back());
  rewriter.create<mlir::simulation::YieldOp>(modelOp.getLoc(), std::nullopt);

  llvm::SmallVector<VariableGetOp> variableGetOps;

  initFunctionOp->walk([&](VariableGetOp getOp) {
    variableGetOps.push_back(getOp);
  });

  for (VariableGetOp getOp : variableGetOps) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    rewriter.setInsertionPoint(getOp);
    rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(getOp, variableOp);
  }

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createDeinitFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto deinitFunctionOp =
      builder.create<mlir::simulation::DeinitFunctionOp>(modelOp.getLoc());

  mlir::Block* entryBlock =
      builder.createBlock(&deinitFunctionOp.getBodyRegion());

  builder.setInsertionPointToStart(entryBlock);
  builder.create<mlir::simulation::YieldOp>(modelOp.getLoc(), std::nullopt);

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::declareGlobalVariables(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<VariableOp> variables,
    llvm::StringMap<GlobalVariableOp>& globalVariablesMap)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  for (VariableOp variable : variables) {
    auto globalVariableOp = builder.create<GlobalVariableOp>(
        variable.getLoc(), "var",
        variable.getVariableType().toArrayType());

    symbolTableCollection.getSymbolTable(moduleOp).insert(
        globalVariableOp, moduleOp.getBody()->begin());

    globalVariablesMap[variable.getSymName()] = globalVariableOp;
  }

  return mlir::success();
}

namespace
{
  template<typename Op>
  class QualifiedVariableOpPattern : public mlir::OpRewritePattern<Op>
  {
    public:
      QualifiedVariableOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::StringMap<GlobalVariableOp>& globalVariablesMap)
          : mlir::OpRewritePattern<Op>(context),
            symbolTableCollection(&symbolTableCollection),
            globalVariablesMap(&globalVariablesMap)
      {
      }

    protected:
      mlir::SymbolTableCollection& getSymbolTableCollection() const
      {
        assert(symbolTableCollection != nullptr);
        return *symbolTableCollection;
      }

      [[nodiscard]] GlobalVariableOp getGlobalVariable(
          VariableOp variableOp) const
      {
        assert(globalVariablesMap != nullptr);
        assert(globalVariablesMap->contains(variableOp.getSymName()));
        return (*globalVariablesMap)[variableOp.getSymName()];
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      llvm::StringMap<GlobalVariableOp>* globalVariablesMap;
  };

  class QualifiedVariableGetOpPattern
      : public QualifiedVariableOpPattern<QualifiedVariableGetOp>
  {
    public:
      using QualifiedVariableOpPattern<QualifiedVariableGetOp>
          ::QualifiedVariableOpPattern;

      mlir::LogicalResult matchAndRewrite(
          QualifiedVariableGetOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

        mlir::Value replacement = rewriter.create<GlobalVariableGetOp>(
            op.getLoc(), getGlobalVariable(variableOp));

        if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
            arrayType && arrayType.isScalar()) {
          replacement = rewriter.create<LoadOp>(
              replacement.getLoc(), replacement, std::nullopt);
        }

        rewriter.replaceOp(op, replacement);
        return mlir::success();
      }
  };

  class QualifiedVariableSetOpPattern
      : public QualifiedVariableOpPattern<QualifiedVariableSetOp>
  {
    public:
    using QualifiedVariableOpPattern<QualifiedVariableSetOp>
        ::QualifiedVariableOpPattern;

    mlir::LogicalResult matchAndRewrite(
        QualifiedVariableSetOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

      mlir::Value globalVariable = rewriter.create<GlobalVariableGetOp>(
          op.getLoc(), getGlobalVariable(variableOp));

      mlir::Value storedValue = op.getValue();
      auto arrayType = globalVariable.getType().cast<ArrayType>();

      if (!arrayType.isScalar()) {
        return mlir::failure();
      }

      if (mlir::Type expectedType = arrayType.getElementType();
          storedValue.getType() != expectedType) {
        storedValue = rewriter.create<CastOp>(
            storedValue.getLoc(), expectedType, storedValue);
      }

      rewriter.replaceOpWithNewOp<StoreOp>(
          op, storedValue, globalVariable, std::nullopt);

      return mlir::success();
    }
  };
}

mlir::LogicalResult
ModelicaToSimulationConversionPass::convertQualifiedVariableAccesses(
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::StringMap<GlobalVariableOp>& globalVariablesMap)
{
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<QualifiedVariableGetOp>(
            [&](QualifiedVariableGetOp op) {
              VariableOp variableOp = op.getVariableOp(symbolTableCollection);
              auto parentModelOp = variableOp->getParentOfType<ModelOp>();

              if (!parentModelOp || parentModelOp != modelOp) {
                return true;
              }

              return false;
            });

target.addDynamicallyLegalOp<QualifiedVariableSetOp>(
    [&](QualifiedVariableSetOp op) {
      VariableOp variableOp = op.getVariableOp(symbolTableCollection);
      auto parentModelOp = variableOp->getParentOfType<ModelOp>();

      if (!parentModelOp || parentModelOp != modelOp) {
        return true;
      }

      return false;
    });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<
      QualifiedVariableGetOpPattern,
      QualifiedVariableSetOpPattern>(&getContext(), symbolTableCollection,
                                     globalVariablesMap);

  return mlir::applyPartialConversion(moduleOp, target, std::move(patterns));
}

GlobalVariableOp ModelicaToSimulationConversionPass::declareTimeVariable(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto timeType = RealType::get(builder.getContext());

  auto globalVariableOp = builder.create<GlobalVariableOp>(
      moduleOp.getLoc(), "time", ArrayType::get(std::nullopt, timeType));

  symbolTableCollection.getSymbolTable(moduleOp).insert(globalVariableOp);
  return globalVariableOp;
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createTimeGetterOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    GlobalVariableOp timeVariableOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = timeVariableOp.getLoc();

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "getTime",
      builder.getFunctionType(std::nullopt, builder.getF64Type()));

  symbolTableCollection.getSymbolTable(moduleOp).insert(functionOp);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value array =
      builder.create<GlobalVariableGetOp>(loc, timeVariableOp);

  mlir::Value result = builder.create<LoadOp>(
      loc, RealType::get(builder.getContext()), array);

  result = builder.create<CastOp>(loc, builder.getF64Type(), result);
  builder.create<mlir::simulation::ReturnOp>(loc, result);

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createTimeSetterOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    GlobalVariableOp timeVariableOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = timeVariableOp.getLoc();

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "setTime",
      builder.getFunctionType(builder.getF64Type(), std::nullopt));

  symbolTableCollection.getSymbolTable(moduleOp).insert(functionOp);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value array =
      builder.create<GlobalVariableGetOp>(loc, timeVariableOp);

  mlir::Value newTime = builder.create<CastOp>(
      loc, RealType::get(builder.getContext()), functionOp.getArgument(0));

  builder.create<StoreOp>(loc, newTime, array, std::nullopt);
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);

  return mlir::success();
}

namespace
{
  class TimeOpLowering : public mlir::OpRewritePattern<TimeOp>
  {
    public:
    using mlir::OpRewritePattern<TimeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        TimeOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto timeType = op.getType();

      auto globalGetOp = rewriter.create<GlobalVariableGetOp>(
          op.getLoc(),
          ArrayType::get(std::nullopt, timeType),
          "time");

      rewriter.replaceOpWithNewOp<LoadOp>(
          op, timeType, globalGetOp, std::nullopt);

      return mlir::success();
    }
  };
}

mlir::LogicalResult ModelicaToSimulationConversionPass::convertTimeOp(
    mlir::ModuleOp moduleOp)
{
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<TimeOpLowering>(moduleOp.getContext());
  return mlir::applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToSimulationConversionPass()
  {
    return std::make_unique<ModelicaToSimulationConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToSimulationConversionPass(
      const ModelicaToSimulationConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToSimulationConversionPass>(options);
  }
}
