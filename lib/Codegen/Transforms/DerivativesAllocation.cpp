#include "marco/Codegen/Transforms/DerivativesAllocation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mutex>

namespace mlir::modelica
{
#define GEN_PASS_DEF_DERIVATIVESALLOCATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  struct MutexCollection
  {
    std::mutex symbolTableCollectionMutex;
    std::mutex derivativesMutex;
  };

  class DerivativesAllocationPass
      : public impl::DerivativesAllocationPassBase<DerivativesAllocationPass>
  {
    public:
      using DerivativesAllocationPassBase::DerivativesAllocationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);

      DerivativesMap& getDerivativesMap();

      mlir::LogicalResult collectDerivedIndices(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          EquationInstanceOp equationInstanceOp) const;

      mlir::LogicalResult collectDerivedIndices(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          AlgorithmOp algorithmOp) const;

      mlir::LogicalResult collectDerivedIndices(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          InitialAlgorithmOp initialAlgorithmOp) const;

      mlir::LogicalResult collectDerivedIndicesInAlgorithmRegion(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          mlir::Region& region) const;

      mlir::LogicalResult createDerivativeVariables(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          DerivativesMap& derivativesMap,
          const llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          MutexCollection& mutexCollection);

      mlir::LogicalResult removeDerOps(
          mlir::SymbolTableCollection& symbolTableCollection,
          const DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          EquationInstanceOp equationInstanceOp);

      mlir::LogicalResult removeDerOps(
          mlir::SymbolTableCollection& symbolTableCollection,
          const DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          AlgorithmOp algorithmOp);

      mlir::LogicalResult removeDerOps(
          mlir::SymbolTableCollection& symbolTableCollection,
          const DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection,
          InitialAlgorithmOp initialAlgorithmOp);

      mlir::LogicalResult createStartOpsAndDummyEquations(
          ModelOp modelOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          const llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
          const DerivativesMap& derivativesMap,
          MutexCollection& mutexCollection);
  };
}

void DerivativesAllocationPass::runOnOperation()
{
  if (mlir::failed(processModelOp(getOperation()))) {
    return signalPassFailure();
  }

  markAnalysesPreserved<DerivativesMap>();
}

mlir::LogicalResult DerivativesAllocationPass::processModelOp(ModelOp modelOp)
{
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::DenseSet<EquationInstanceOp> equationInstanceOps;
  llvm::DenseSet<AlgorithmOp> algorithmOps;
  llvm::DenseSet<InitialAlgorithmOp> initialAlgorithmOps;

  for (auto& op : modelOp.getOps()) {
    if (auto equationInstanceOp = mlir::dyn_cast<EquationInstanceOp>(op)) {
      equationInstanceOps.insert(equationInstanceOp);
      continue;
    }

    if (auto algorithmOp = mlir::dyn_cast<AlgorithmOp>(op)) {
      algorithmOps.insert(algorithmOp);
      continue;
    }

    if (auto initialAlgorithmOp = mlir::dyn_cast<InitialAlgorithmOp>(op)) {
      initialAlgorithmOps.insert(initialAlgorithmOp);
      continue;
    }
  }

  // Collect the derived indices.
  llvm::DenseSet<mlir::SymbolRefAttr> derivedVariables;
  DerivativesMap& derivativesMap = getDerivativesMap();
  MutexCollection mutexCollection;

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationInstanceOps,
          [&](EquationInstanceOp equation) {
            return collectDerivedIndices(
                modelOp, symbolTableCollection,
                derivedVariables, derivativesMap, mutexCollection, equation);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps,
          [&](AlgorithmOp algorithmOp) {
            return collectDerivedIndices(
                modelOp, symbolTableCollection,
                derivedVariables, derivativesMap, mutexCollection,
                algorithmOp);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), initialAlgorithmOps,
          [&](InitialAlgorithmOp initialAlgorithmOp) {
            return collectDerivedIndices(
                modelOp, symbolTableCollection,
                derivedVariables, derivativesMap, mutexCollection,
                initialAlgorithmOp);
          }))) {
    return mlir::failure();
  }

  // Create the derivative variables.
  if (mlir::failed(createDerivativeVariables(
          modelOp, symbolTableCollection, derivativesMap, derivedVariables,
          mutexCollection))) {
    return mlir::failure();
  }

  // Replace the derivative operations.
  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationInstanceOps,
          [&](EquationInstanceOp equation) {
            return removeDerOps(
                symbolTableCollection, derivativesMap, mutexCollection,
                equation);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps,
          [&](AlgorithmOp algorithmOp) {
            return removeDerOps(
                symbolTableCollection, derivativesMap, mutexCollection,
                algorithmOp);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), initialAlgorithmOps,
          [&](InitialAlgorithmOp initialAlgorithmOp) {
            return removeDerOps(
                symbolTableCollection, derivativesMap, mutexCollection,
                initialAlgorithmOp);
          }))) {
    return mlir::failure();
  }

  // Create the start values for all the indices and the equations for the
  // indices that are not derived.
  if (mlir::failed(createStartOpsAndDummyEquations(
          modelOp, symbolTableCollection, derivedVariables, derivativesMap,
          mutexCollection))) {
    return mlir::failure();
  }

  return mlir::success();
}

DerivativesMap& DerivativesAllocationPass::getDerivativesMap()
{
  if (auto analysis = getCachedAnalysis<DerivativesMap>()) {
    return *analysis;
  }

  auto& analysis = getAnalysis<DerivativesMap>();
  analysis.initialize();
  return analysis;
}

static mlir::SymbolRefAttr getSymbolRefFromPath(
    llvm::ArrayRef<mlir::FlatSymbolRefAttr> symbols)
{
  assert(!symbols.empty());
  return mlir::SymbolRefAttr::get(symbols[0].getAttr(), symbols.drop_front());
}

static mlir::LogicalResult getShape(
    llvm::SmallVectorImpl<int64_t>& shape,
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    std::mutex& symbolTableMutex,
    mlir::SymbolRefAttr variable)
{
  std::lock_guard<std::mutex> lock(symbolTableMutex);
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, variable.getRootReference());

  if (!variableOp) {
    return mlir::failure();
  }

  auto variableShape = variableOp.getVariableType().getShape();
  shape.append(variableShape.begin(), variableShape.end());

  for (mlir::FlatSymbolRefAttr component : variable.getNestedReferences()) {
    assert(variableOp.getVariableType().unwrap().isa<RecordType>());

    auto recordOp = mlir::cast<RecordOp>(
        variableOp.getVariableType().unwrap().cast<RecordType>()
            .getRecordOp(symbolTableCollection, moduleOp));

    variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        recordOp, component.getAttr());

    if (!variableOp) {
      return mlir::failure();
    }

    auto componentShape = variableOp.getVariableType().getShape();
    shape.append(componentShape.begin(), componentShape.end());
  }

  return mlir::success();
}

static IndexSet shapeToIndexSet(llvm::ArrayRef<int64_t> shape)
{
  IndexSet result;
  llvm::SmallVector<Range, 3> ranges;

  for (int64_t dimension : shape) {
    ranges.push_back(Range(0, dimension));
  }

  result += MultidimensionalRange(std::move(ranges));
  return result;
}

static mlir::LogicalResult getAccess(
    mlir::Value value,
    llvm::SmallVectorImpl<mlir::FlatSymbolRefAttr>& symbols,
    llvm::SmallVectorImpl<mlir::Value>& indices)
{
  mlir::Operation* definingOp = value.getDefiningOp();

  if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(definingOp)) {
    symbols.push_back(
        mlir::FlatSymbolRefAttr::get(variableGetOp.getVariableAttr()));

    std::reverse(symbols.begin(), symbols.end());
    std::reverse(indices.begin(), indices.end());

    return mlir::success();
  }

  if (auto componentGetOp = mlir::dyn_cast<ComponentGetOp>(definingOp)) {
    symbols.push_back(
        mlir::FlatSymbolRefAttr::get(componentGetOp.getComponentNameAttr()));

    return getAccess(componentGetOp.getVariable(), symbols, indices);
  }

  if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
    for (size_t i = 0, e = loadOp.getIndices().size(); i < e; ++i) {
      indices.push_back(loadOp.getIndices()[e - i - 1]);
    }

    return getAccess(loadOp.getArray(), symbols, indices);
  }

  if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
    for (size_t i = 0, e = subscriptionOp.getIndices().size(); i < e; ++i) {
      indices.push_back(subscriptionOp.getIndices()[e - i - 1]);
    }

    return getAccess(subscriptionOp.getSource(), symbols, indices);
  }

  return mlir::failure();
}

mlir::LogicalResult DerivativesAllocationPass::collectDerivedIndices(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    EquationInstanceOp equationInstanceOp) const
{
  llvm::SmallVector<DerOp> derOps;
  EquationTemplateOp equationTemplateOp = equationInstanceOp.getTemplate();

  equationTemplateOp.getBody()->walk([&](DerOp derOp) {
    derOps.push_back(derOp);
  });

  for (DerOp derOp : derOps) {
    llvm::SmallVector<mlir::FlatSymbolRefAttr, 3> symbols;
    llvm::SmallVector<mlir::Value, 3> indices;

    if (mlir::failed(getAccess(derOp.getOperand(), symbols, indices))) {
      derOp.emitOpError() << "Can't obtain the access to the variable";
      return mlir::failure();
    }

    mlir::SymbolRefAttr variable = getSymbolRefFromPath(symbols);
    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(
            variableShape, modelOp, symbolTableCollection,
            mutexCollection.symbolTableCollectionMutex,
            variable))) {
      return mlir::failure();
    }

    auto affineMap = equationTemplateOp.getAccessFunction(indices);

    if (!affineMap) {
      derOp.emitOpError() << "Can't analyze the access to the variable";
      return mlir::failure();
    }

    IndexSet derivedIndices;

    if (affineMap->getNumResults() != 0) {
      auto accessFunction = AccessFunction::build(*affineMap);
      IndexSet equationIndices;

      if (auto equationRanges = equationInstanceOp.getIndices()) {
        equationIndices += equationRanges->getValue();
      }

      derivedIndices = accessFunction->map(equationIndices);
    }

    size_t derivedIndicesRank = derivedIndices.rank();
    size_t variableIndicesRank = variableShape.size();

    if (derivedIndicesRank < variableIndicesRank) {
      llvm::SmallVector<Range, 3> extraDimensions;

      for (size_t i = derivedIndicesRank; i < variableIndicesRank; ++i) {
        extraDimensions.push_back(Range(0, variableShape[i]));
      }

      derivedIndices = derivedIndices.append(
          IndexSet(MultidimensionalRange(extraDimensions)));
    }

    // Add the derived indices.
    std::lock_guard<std::mutex> lock(mutexCollection.derivativesMutex);
    derivedVariables.insert(variable);
    derivativesMap.addDerivedIndices(variable, std::move(derivedIndices));
  }

  return mlir::success();
}

mlir::LogicalResult DerivativesAllocationPass::collectDerivedIndices(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    AlgorithmOp algorithmOp) const
{
  return collectDerivedIndicesInAlgorithmRegion(
      modelOp, symbolTableCollection,
      derivedVariables, derivativesMap, mutexCollection,
      algorithmOp.getBodyRegion());
}

mlir::LogicalResult DerivativesAllocationPass::collectDerivedIndices(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    InitialAlgorithmOp initialAlgorithmOp) const
{
  return collectDerivedIndicesInAlgorithmRegion(
      modelOp, symbolTableCollection,
      derivedVariables, derivativesMap, mutexCollection,
      initialAlgorithmOp.getBodyRegion());
}

mlir::LogicalResult
DerivativesAllocationPass::collectDerivedIndicesInAlgorithmRegion(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    mlir::Region& region) const
{
  llvm::SmallVector<DerOp> derOps;

  region.walk([&](DerOp derOp) {
    derOps.push_back(derOp);
  });

  for (DerOp derOp : derOps) {
    llvm::SmallVector<mlir::FlatSymbolRefAttr, 3> symbols;
    llvm::SmallVector<mlir::Value, 3> indices;

    if (mlir::failed(getAccess(derOp.getOperand(), symbols, indices))) {
      derOp.emitOpError() << "Can't obtain the access to the variable";
      return mlir::failure();
    }

    mlir::SymbolRefAttr variable = getSymbolRefFromPath(symbols);
    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(
            variableShape, modelOp, symbolTableCollection,
            mutexCollection.symbolTableCollectionMutex,
            variable))) {
      return mlir::failure();
    }

    // Add the derived indices.
    std::lock_guard<std::mutex> lock(mutexCollection.derivativesMutex);
    derivedVariables.insert(variable);
    derivativesMap.addDerivedIndices(variable, shapeToIndexSet(variableShape));
  }

  return mlir::success();
}

static std::string getDerivativeName(mlir::SymbolRefAttr variableName)
{
  std::string result = "der_" + variableName.getRootReference().str();

  for (mlir::FlatSymbolRefAttr component :
       variableName.getNestedReferences()) {
    result += "." + component.getValue().str();
  }

  return result;
}

mlir::LogicalResult DerivativesAllocationPass::createDerivativeVariables(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    DerivativesMap& derivativesMap,
    const llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    MutexCollection& mutexCollection)
{
  mlir::OpBuilder builder(modelOp);
  builder.setInsertionPointToEnd(modelOp.getBody());

  mlir::SymbolTable& symbolTable =
      symbolTableCollection.getSymbolTable(modelOp);

  llvm::SmallVector<mlir::Attribute> derivativeAttrs;

  // Add the already existing attributes.
  for (mlir::Attribute attr : modelOp.getDerivativesMap()) {
    derivativeAttrs.push_back(attr);
  }

  // Add the new attributes.
  for (mlir::SymbolRefAttr variable : derivedVariables) {
    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(
            variableShape, modelOp, symbolTableCollection,
            mutexCollection.symbolTableCollectionMutex,
            variable))) {
      return mlir::failure();
    }

    auto derVariableOp = builder.create<VariableOp>(
        modelOp.getLoc(), getDerivativeName(variable),
        VariableType::get(variableShape, RealType::get(builder.getContext()),
                          VariabilityProperty::none, IOProperty::none));

    symbolTable.insert(derVariableOp, modelOp.getBody()->end());
    IndexSetAttr derivedIndicesAttr = nullptr;

    if (auto derivedIndices = derivativesMap.getDerivedIndices(variable);
        derivedIndices && !derivedIndices->get().empty()) {
      derivedIndicesAttr = IndexSetAttr::get(
          builder.getContext(),
          derivedIndices->get().getCanonicalRepresentation());
    }

    auto derivative = mlir::SymbolRefAttr::get(derVariableOp.getSymNameAttr());

    derivativeAttrs.push_back(VarDerivativeAttr::get(
        builder.getContext(), variable, derivative, derivedIndicesAttr));

    derivativesMap.setDerivative(variable, derivative);
  }

  modelOp.setDerivativesMapAttr(builder.getArrayAttr(derivativeAttrs));
  return mlir::success();
}

static VariableOp resolveVariable(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::SymbolRefAttr variable)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, variable.getRootReference());

  for (mlir::FlatSymbolRefAttr component : variable.getNestedReferences()) {
    auto recordOp = mlir::cast<RecordOp>(
        variableOp.getVariableType().unwrap().cast<RecordType>()
            .getRecordOp(symbolTableCollection, moduleOp));

    variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        recordOp, component.getAttr());
  }

  return variableOp;
}

namespace
{
  class DerOpRemovePattern : public mlir::OpRewritePattern<DerOp>
  {
    public:
      DerOpRemovePattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          std::mutex& symbolTableMutex,
          const DerivativesMap& derivativesMap)
          : mlir::OpRewritePattern<DerOp>(context),
            symbolTableCollection(&symbolTableCollection),
            symbolTableMutex(&symbolTableMutex),
            derivativesMap(&derivativesMap)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          DerOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();
        auto modelOp = op->getParentOfType<ModelOp>();

        llvm::SmallVector<mlir::FlatSymbolRefAttr, 3> symbols;
        llvm::SmallVector<mlir::Value, 3> indices;

        if (mlir::failed(getAccess(op.getOperand(), symbols, indices))) {
          return mlir::failure();
        }

        mlir::SymbolRefAttr variableName = getSymbolRefFromPath(symbols);
        auto derivativeName = derivativesMap->getDerivative(variableName);

        if (!derivativeName) {
          return mlir::failure();
        }

        VariableOp variableOp = resolveVariable(modelOp, *derivativeName);

        mlir::Value replacement =
            rewriter.create<VariableGetOp>(loc, variableOp);

        if (!indices.empty()) {
          replacement =
              rewriter.create<SubscriptionOp>(loc, replacement, indices);
        }

        if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
            arrayType && arrayType.isScalar()) {
          replacement = rewriter.create<LoadOp>(loc, replacement, std::nullopt);
        }

        rewriter.replaceOp(op, replacement);
        return mlir::success();
      }

    private:
      VariableOp resolveVariable(
          ModelOp modelOp, mlir::SymbolRefAttr variable) const
      {
        std::lock_guard<std::mutex> lock(*symbolTableMutex);
        return ::resolveVariable(modelOp, *symbolTableCollection, variable);
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      std::mutex* symbolTableMutex;
      const DerivativesMap* derivativesMap;
  };
}

mlir::LogicalResult DerivativesAllocationPass::removeDerOps(
    mlir::SymbolTableCollection& symbolTableCollection,
    const DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    EquationInstanceOp equationInstanceOp)
{
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<DerOpRemovePattern>(
      &getContext(), symbolTableCollection,
      mutexCollection.symbolTableCollectionMutex, derivativesMap);

  return applyPatternsAndFoldGreedily(
      equationInstanceOp.getTemplate(), std::move(patterns));
}

mlir::LogicalResult DerivativesAllocationPass::removeDerOps(
    mlir::SymbolTableCollection& symbolTableCollection,
    const DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    AlgorithmOp algorithmOp)
{
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<DerOpRemovePattern>(
      &getContext(), symbolTableCollection,
      mutexCollection.symbolTableCollectionMutex, derivativesMap);

  return applyPatternsAndFoldGreedily(algorithmOp, std::move(patterns));
}

mlir::LogicalResult DerivativesAllocationPass::removeDerOps(
    mlir::SymbolTableCollection& symbolTableCollection,
    const DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection,
    InitialAlgorithmOp initialAlgorithmOp)
{
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<DerOpRemovePattern>(
      &getContext(), symbolTableCollection,
      mutexCollection.symbolTableCollectionMutex, derivativesMap);

  return applyPatternsAndFoldGreedily(initialAlgorithmOp, std::move(patterns));
}

static mlir::LogicalResult createStartOp(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    mlir::SymbolRefAttr variable)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modelOp.getBody());

  VariableOp variableOp =
      resolveVariable(modelOp, symbolTableCollection, variable);

  mlir::Location loc = variableOp.getLoc();

  // TODO handle object orientation
  auto startOp = builder.create<StartOp>(
      loc, variable.getRootReference().getValue(), false, false);

  assert(startOp.getBodyRegion().empty());
  mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
  builder.setInsertionPointToStart(bodyBlock);

  mlir::Value zero = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 0));

  VariableType variableType = variableOp.getVariableType();

  if (!variableType.isScalar()) {
    zero = builder.create<ArrayBroadcastOp>(
          loc, variableType.toArrayType(), zero);
  }

  builder.create<YieldOp>(loc, zero);
  return mlir::success();
}

static mlir::LogicalResult createInitialEquations(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    mlir::SymbolRefAttr derivativeName,
    const IndexSet& indices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modelOp.getBody());

  VariableOp variableOp =
      resolveVariable(modelOp, symbolTableCollection, derivativeName);

  mlir::Location loc = variableOp.getLoc();

  auto equationTemplateOp = builder.create<EquationTemplateOp>(loc);

  builder.setInsertionPointToStart(
      equationTemplateOp.createBody(indices.rank()));

  mlir::Value variable = builder.create<VariableGetOp>(loc, variableOp);

  variable = builder.create<LoadOp>(
      loc, variable, equationTemplateOp.getInductionVariables());

  mlir::Value zero = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value lhs = builder.create<EquationSideOp>(loc, variable);
  mlir::Value rhs = builder.create<EquationSideOp>(loc, zero);
  builder.create<EquationSidesOp>(loc, lhs, rhs);

  builder.setInsertionPointAfter(equationTemplateOp);

  for (const MultidimensionalRange& range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    auto instanceOp = builder.create<EquationInstanceOp>(
        loc, equationTemplateOp, false);

    instanceOp.setIndicesAttr(
        MultidimensionalRangeAttr::get(builder.getContext(), range));
  }

  return mlir::success();
}

mlir::LogicalResult
DerivativesAllocationPass::createStartOpsAndDummyEquations(
    ModelOp modelOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    const llvm::DenseSet<mlir::SymbolRefAttr>& derivedVariables,
    const DerivativesMap& derivativesMap,
    MutexCollection& mutexCollection)
{
  mlir::OpBuilder builder(modelOp);

  for (mlir::SymbolRefAttr variableName : derivedVariables) {
    auto derivativeName = derivativesMap.getDerivative(variableName);

    if (!derivativeName) {
      continue;
    }

    // Create the start value.
    if (mlir::failed(createStartOp(
            builder, symbolTableCollection, modelOp, *derivativeName))) {
      return mlir::failure();
    }

    // Create the equations for the non-derived indices.
    llvm::SmallVector<int64_t, 3> variableDimensions;

    if (mlir::failed(getShape(
            variableDimensions, modelOp, symbolTableCollection,
            mutexCollection.symbolTableCollectionMutex,
            variableName))) {
      return mlir::failure();
    }

    if (!variableDimensions.empty()) {
      IndexSet nonDerivedIndices = shapeToIndexSet(variableDimensions);

      if (auto derivedIndices =
              derivativesMap.getDerivedIndices(variableName)) {
        nonDerivedIndices -= derivedIndices->get();
      }

      if (!nonDerivedIndices.empty()) {
        if (mlir::failed(createInitialEquations(
                builder, symbolTableCollection, modelOp,
                *derivativeName,
                nonDerivedIndices.getCanonicalRepresentation()))) {
          return mlir::failure();
        }
      }
    }
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createDerivativesAllocationPass()
  {
    return std::make_unique<DerivativesAllocationPass>();
  }
}
