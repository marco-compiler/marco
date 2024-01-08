#include "marco/Codegen/Conversion/ModelicaToSimulation/ModelicaToSimulation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/VariableFilter/VariableFilter.h"
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

      mlir::LogicalResult processModelOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<SimulationVariableOp> variables,
          ModelOp modelOp);

      mlir::LogicalResult createModelNameOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult createNumOfVariablesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      mlir::LogicalResult createVariableNamesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      mlir::LogicalResult createVariableRanksOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      mlir::LogicalResult createPrintableIndicesOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables,
          const marco::VariableFilter& variablesFilter);

      mlir::LogicalResult createDerivativesMapOp(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      mlir::LogicalResult createVariableGetters(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      /// Create the function that is called before starting the simulation.
      mlir::LogicalResult createInitFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SimulationVariableOp> variables);

      /// Create the function that is called when the simulation has finished.
      mlir::LogicalResult createDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult convertSimulationVarsToGlobalVars(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp);

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

  llvm::SmallVector<SimulationVariableOp> variables;

  for (SimulationVariableOp variable :
       moduleOp.getOps<SimulationVariableOp>()) {
    variables.push_back(variable);
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
          rewriter, symbolTableCollection, moduleOp, variables, modelOps[0]))) {
    return signalPassFailure();
  }

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

mlir::LogicalResult ModelicaToSimulationConversionPass::processModelOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<SimulationVariableOp> variables,
    ModelOp modelOp)
{
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

  if (mlir::failed(convertSimulationVarsToGlobalVars(
          rewriter, symbolTableCollection, moduleOp))) {
    return mlir::failure();
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
    llvm::ArrayRef<SimulationVariableOp> variables)
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
    llvm::ArrayRef<SimulationVariableOp> variable)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<llvm::StringRef> names;

  for (SimulationVariableOp variableOp : variable) {
    names.push_back(variableOp.getSymName());
  }

  builder.create<mlir::simulation::VariableNamesOp>(
      modelOp.getLoc(), builder.getStrArrayAttr(names));

  return mlir::success();
}

mlir::LogicalResult ModelicaToSimulationConversionPass::createVariableRanksOp(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SimulationVariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<int64_t> ranks;

  for (SimulationVariableOp variableOp : variables) {
    VariableType variableType = variableOp.getVariableType();
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
    llvm::ArrayRef<SimulationVariableOp> variables,
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

  for (SimulationVariableOp variableOp : variables) {
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
    llvm::ArrayRef<SimulationVariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Map the position of the variables for faster lookups.
  llvm::DenseMap<mlir::SymbolRefAttr, size_t> positionsMap;

  for (size_t i = 0, e = variables.size(); i < e; ++i) {
    SimulationVariableOp variableOp = variables[i];
    auto name = mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr());
    positionsMap[name] = i;
  }

  // Compute the positions of the derivatives.
  llvm::SmallVector<int64_t> derivatives;
  auto& derivativesMap = getDerivativesMap(modelOp);

  for (SimulationVariableOp variableOp : variables) {
    if (auto derivative = derivativesMap.getDerivative(
            mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
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
    llvm::ArrayRef<SimulationVariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Create a getter for each variable.
  size_t variableGetterCounter = 0;
  llvm::SmallVector<mlir::Attribute> getterNames;

  for (SimulationVariableOp simulationVariable : variables) {
    builder.setInsertionPointToEnd(moduleOp.getBody());
    VariableType variableType = simulationVariable.getVariableType();

    auto getterOp = builder.create<mlir::simulation::VariableGetterOp>(
        simulationVariable.getLoc(),
        "var_getter_" + std::to_string(variableGetterCounter++),
        variableType.getRank());

    getterNames.push_back(
        mlir::FlatSymbolRefAttr::get(getterOp.getSymNameAttr()));

    mlir::Block* bodyBlock = getterOp.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    mlir::Value getOp = builder.create<SimulationVariableGetOp>(
        simulationVariable.getLoc(), simulationVariable);

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
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SimulationVariableOp> variables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto initFunctionOp =
      builder.create<mlir::simulation::InitFunctionOp>(modelOp.getLoc());

  mlir::Block* entryBlock =
      builder.createBlock(&initFunctionOp.getBodyRegion());

  builder.setInsertionPointToStart(entryBlock);

  // Keep track of the variables for which a start value has been provided.
  llvm::DenseSet<llvm::StringRef> initializedVars;

  for (StartOp startOp : modelOp.getOps<StartOp>()) {
    // Set the variable as initialized.
    initializedVars.insert(startOp.getVariable());

    // Note that read-only variables must be set independently of the 'fixed'
    // attribute being true or false.

    auto simulationVariableOp =
        symbolTableCollection.lookupSymbolIn<SimulationVariableOp>(
            moduleOp, startOp.getVariableAttr());

    if (!simulationVariableOp) {
      startOp.emitError() << "simulation variable not found";
      return mlir::failure();
    }

    if (startOp.getFixed() && !simulationVariableOp.isReadOnly()) {
      continue;
    }

    mlir::IRMapping startOpsMapping;

    for (auto& op : startOp.getBodyRegion().getOps()) {
      if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
        mlir::Value valueToBeStored =
            startOpsMapping.lookup(yieldOp.getValues()[0]);

        if (startOp.getEach()) {
          if (simulationVariableOp.getVariableType().isScalar()) {
            builder.create<SimulationVariableSetOp>(
                startOp.getLoc(), simulationVariableOp, valueToBeStored);
          } else {
            mlir::Value destination = builder.create<SimulationVariableGetOp>(
                startOp.getLoc(), simulationVariableOp);

            builder.create<ArrayFillOp>(
                startOp.getLoc(), destination, valueToBeStored);
          }
        } else {
          auto valueType = valueToBeStored.getType();

          if (auto valueArrayType = valueType.dyn_cast<ArrayType>()) {
            mlir::Value destination = builder.create<SimulationVariableGetOp>(
                startOp.getLoc(), simulationVariableOp);

            builder.create<ArrayCopyOp>(
                startOp.getLoc(), valueToBeStored, destination);
          } else {
            builder.create<SimulationVariableSetOp>(
                startOp.getLoc(), simulationVariableOp, valueToBeStored);
          }
        }
      } else {
        builder.clone(op, startOpsMapping);
      }
    }
  }

  // The variables without a 'start' attribute must be initialized to zero.
  for (SimulationVariableOp simulationVariable : variables) {
    if (initializedVars.contains(simulationVariable.getSymName())) {
      continue;
    }

    VariableType variableType = simulationVariable.getVariableType();

    auto zeroMaterializableElementType =
        variableType.getElementType().dyn_cast<ZeroMaterializableType>();

    if (!zeroMaterializableElementType) {
      return mlir::failure();
    }

    mlir::Value zeroValue =
        zeroMaterializableElementType.materializeZeroValuedConstant(
            builder, simulationVariable.getLoc());

    if (variableType.isScalar()) {
      builder.create<SimulationVariableSetOp>(
          simulationVariable.getLoc(), simulationVariable, zeroValue);
    } else {
      mlir::Value destination = builder.create<SimulationVariableGetOp>(
          simulationVariable.getLoc(), simulationVariable);

      builder.create<ArrayFillOp>(destination.getLoc(), destination, zeroValue);
    }
  }

  builder.setInsertionPointToEnd(&initFunctionOp.getBodyRegion().back());
  builder.create<mlir::simulation::YieldOp>(modelOp.getLoc(), std::nullopt);

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

mlir::LogicalResult ModelicaToSimulationConversionPass
    ::convertSimulationVarsToGlobalVars(
        mlir::RewriterBase& rewriter,
        mlir::SymbolTableCollection& symbolTableCollection,
        mlir::ModuleOp moduleOp)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  llvm::StringMap<GlobalVariableOp> simulationToGlobalVariablesMap;
  llvm::SmallVector<SimulationVariableOp> simulationVariables;

  for (SimulationVariableOp simulationVariable :
       moduleOp.getOps<SimulationVariableOp>()) {
    simulationVariables.push_back(simulationVariable);
  }

  for (SimulationVariableOp simulationVariable : simulationVariables) {
    rewriter.setInsertionPoint(simulationVariable);

    auto globalVariableOp = rewriter.replaceOpWithNewOp<GlobalVariableOp>(
        simulationVariable, "var",
        simulationVariable.getVariableType().toArrayType());

    symbolTableCollection.getSymbolTable(moduleOp).insert(
        globalVariableOp, moduleOp.getBody()->begin());

    simulationToGlobalVariablesMap[simulationVariable.getSymName()] =
        globalVariableOp;
  }

  llvm::SmallVector<SimulationVariableGetOp> getOps;
  llvm::SmallVector<SimulationVariableSetOp> setOps;

  moduleOp.walk([&](mlir::Operation* nestedOp) {
    if (auto getOp = mlir::dyn_cast<SimulationVariableGetOp>(nestedOp)) {
      getOps.push_back(getOp);
    }

    if (auto setOp = mlir::dyn_cast<SimulationVariableSetOp>(nestedOp)) {
      setOps.push_back(setOp);
    }
  });

  for (SimulationVariableGetOp getOp : getOps) {
    rewriter.setInsertionPoint(getOp);

    mlir::Value replacement = rewriter.create<GlobalVariableGetOp>(
        getOp.getLoc(), simulationToGlobalVariablesMap[getOp.getVariable()]);

    if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
        arrayType && arrayType.isScalar()) {
      replacement = rewriter.create<LoadOp>(
          replacement.getLoc(), replacement, std::nullopt);
    }

    rewriter.replaceOp(getOp, replacement);
  }

  for (SimulationVariableSetOp setOp : setOps) {
    rewriter.setInsertionPoint(setOp);

    GlobalVariableOp globalVariableOp =
        simulationToGlobalVariablesMap[setOp.getVariable()];

    mlir::Value globalVariable =
        rewriter.create<GlobalVariableGetOp>(setOp.getLoc(), globalVariableOp);

    mlir::Value storedValue = setOp.getValue();
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
        setOp, storedValue, globalVariable, std::nullopt);
  }

  return mlir::success();
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

  patterns.add<
      TimeOpLowering>(moduleOp.getContext());

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
