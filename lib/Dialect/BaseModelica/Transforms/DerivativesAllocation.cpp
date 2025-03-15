#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivativesMaterialization.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mutex>

namespace mlir::bmodelica {
#define GEN_PASS_DEF_DERIVATIVESMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
struct MutexCollection {
  std::mutex symbolTableCollectionMutex;
  std::mutex derivativesMutex;
};

class DerivativesMaterializationPass
    : public impl::DerivativesMaterializationPassBase<
          DerivativesMaterializationPass> {
public:
  using DerivativesMaterializationPassBase<
      DerivativesMaterializationPass>::DerivativesMaterializationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult collectDerivedIndices(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
      llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
      DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
      EquationInstanceOp equationInstanceOp) const;

  mlir::LogicalResult collectDerivedIndices(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
      llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
      DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
      AlgorithmOp algorithmOp) const;

  mlir::LogicalResult collectDerivedIndicesInAlgorithmRegion(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
      llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
      DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
      mlir::Region &region) const;

  mlir::LogicalResult createDerivativeVariables(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
      DerivativesMap &derivativesMap,
      const llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
      MutexCollection &mutexCollection);

  mlir::LogicalResult
  removeDerOps(mlir::SymbolTableCollection &symbolTableCollection,
               const DerivativesMap &derivativesMap,
               MutexCollection &mutexCollection,
               EquationInstanceOp equationInstanceOp);

  mlir::LogicalResult
  removeDerOps(mlir::SymbolTableCollection &symbolTableCollection,
               const DerivativesMap &derivativesMap,
               MutexCollection &mutexCollection, AlgorithmOp algorithmOp);

  mlir::LogicalResult createStartOpsAndDummyEquations(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
      const llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
      const DerivativesMap &derivativesMap, MutexCollection &mutexCollection);
};
} // namespace

void DerivativesMaterializationPass::runOnOperation() {
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

  markAnalysesPreserved<DerivativesMap>();
}

mlir::LogicalResult
DerivativesMaterializationPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<EquationInstanceOp> equationInstanceOps;
  llvm::SmallVector<AlgorithmOp> algorithmOps;

  modelOp.collectInitialEquations(equationInstanceOps);
  modelOp.collectMainEquations(equationInstanceOps);

  modelOp.collectInitialAlgorithms(algorithmOps);
  modelOp.collectMainAlgorithms(algorithmOps);

  // Collect the derived indices.
  llvm::DenseSet<mlir::SymbolRefAttr> derivedVariables;
  DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  MutexCollection mutexCollection;

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationInstanceOps, [&](EquationInstanceOp equation) {
            return collectDerivedIndices(modelOp, symbolTableCollection,
                                         derivedVariables, derivativesMap,
                                         mutexCollection, equation);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps, [&](AlgorithmOp algorithmOp) {
            return collectDerivedIndices(modelOp, symbolTableCollection,
                                         derivedVariables, derivativesMap,
                                         mutexCollection, algorithmOp);
          }))) {
    return mlir::failure();
  }

  // Create the derivative variables.
  if (mlir::failed(createDerivativeVariables(modelOp, symbolTableCollection,
                                             derivativesMap, derivedVariables,
                                             mutexCollection))) {
    return mlir::failure();
  }

  // Replace the derivative operations.
  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationInstanceOps, [&](EquationInstanceOp equation) {
            return removeDerOps(symbolTableCollection, derivativesMap,
                                mutexCollection, equation);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps, [&](AlgorithmOp algorithmOp) {
            return removeDerOps(symbolTableCollection, derivativesMap,
                                mutexCollection, algorithmOp);
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

static mlir::SymbolRefAttr
getSymbolRefFromPath(llvm::ArrayRef<mlir::FlatSymbolRefAttr> symbols) {
  assert(!symbols.empty());
  return mlir::SymbolRefAttr::get(symbols[0].getAttr(), symbols.drop_front());
}

static mlir::LogicalResult
getShape(llvm::SmallVectorImpl<int64_t> &shape, ModelOp modelOp,
         mlir::SymbolTableCollection &symbolTableCollection,
         std::mutex &symbolTableMutex, mlir::SymbolRefAttr variable) {
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
    assert(mlir::isa<RecordType>(variableOp.getVariableType().unwrap()));

    auto recordOp = mlir::cast<RecordOp>(
        mlir::cast<RecordType>(variableOp.getVariableType().unwrap())
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

static IndexSet shapeToIndexSet(llvm::ArrayRef<int64_t> shape) {
  IndexSet result;
  llvm::SmallVector<Range, 3> ranges;

  for (int64_t dimension : shape) {
    ranges.push_back(Range(0, dimension));
  }

  result += MultidimensionalRange(std::move(ranges));
  return result;
}

static mlir::LogicalResult
getAccess(mlir::Value value,
          llvm::SmallVectorImpl<mlir::FlatSymbolRefAttr> &symbols,
          llvm::SmallVectorImpl<mlir::Value> &indices) {
  mlir::Operation *definingOp = value.getDefiningOp();

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

  if (auto extractOp = mlir::dyn_cast<TensorExtractOp>(definingOp)) {
    for (size_t i = 0, e = extractOp.getIndices().size(); i < e; ++i) {
      indices.push_back(extractOp.getIndices()[e - i - 1]);
    }

    return getAccess(extractOp.getTensor(), symbols, indices);
  }

  if (auto viewOp = mlir::dyn_cast<TensorViewOp>(definingOp)) {
    for (size_t i = 0, e = viewOp.getSubscriptions().size(); i < e; ++i) {
      indices.push_back(viewOp.getSubscriptions()[e - i - 1]);
    }

    return getAccess(viewOp.getSource(), symbols, indices);
  }

  return mlir::failure();
}

static void
collectDerOps(llvm::SmallVectorImpl<std::pair<DerOp, EquationPath>> &result,
              mlir::Value value, const EquationPath &path) {
  if (auto definingOp = value.getDefiningOp()) {
    if (auto derOp = mlir::dyn_cast<DerOp>(definingOp)) {
      result.emplace_back(derOp, path);
    } else {
      for (unsigned int i = 0, e = definingOp->getNumOperands(); i < e; ++i) {
        collectDerOps(result, definingOp->getOperand(i), path + i);
      }
    }
  }
}

static void
collectDerOps(llvm::SmallVectorImpl<std::pair<DerOp, EquationPath>> &result,
              EquationTemplateOp equationTemplateOp) {
  auto equationSidesOp = mlir::cast<EquationSidesOp>(
      equationTemplateOp.getBody()->getTerminator());

  auto lhsValues = equationSidesOp.getLhsValues();
  auto rhsValues = equationSidesOp.getRhsValues();

  // Left-hand side of the equation.
  for (size_t i = 0, e = lhsValues.size(); i < e; ++i) {
    collectDerOps(result, lhsValues[i], EquationPath(EquationPath::LEFT, i));
  }

  // Right-hand side of the equation.
  for (size_t i = 0, e = lhsValues.size(); i < e; ++i) {
    collectDerOps(result, rhsValues[i], EquationPath(EquationPath::RIGHT, i));
  }
}

mlir::LogicalResult DerivativesMaterializationPass::collectDerivedIndices(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
    DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
    EquationInstanceOp equationInstanceOp) const {
  EquationTemplateOp equationTemplateOp = equationInstanceOp.getTemplate();

  llvm::SmallVector<std::pair<DerOp, EquationPath>> derOps;
  collectDerOps(derOps, equationTemplateOp);

  for (const auto &derOp : derOps) {
    llvm::SmallVector<VariableAccess, 1> accesses;

    {
      std::lock_guard<std::mutex> symbolTableLock(
          mutexCollection.symbolTableCollectionMutex);

      auto access = equationTemplateOp.getAccessAtPath(symbolTableCollection,
                                                       derOp.second + 0);

      if (!access) {
        return mlir::failure();
      }

      accesses.push_back(std::move(*access));
    }

    for (const VariableAccess &access : accesses) {
      llvm::SmallVector<int64_t, 3> variableShape;

      if (mlir::failed(getShape(variableShape, modelOp, symbolTableCollection,
                                mutexCollection.symbolTableCollectionMutex,
                                access.getVariable()))) {
        return mlir::failure();
      }

      const AccessFunction &accessFunction = access.getAccessFunction();

      IndexSet derivedIndices;

      if (accessFunction.getNumOfResults() != 0) {
        derivedIndices =
            accessFunction.map(equationInstanceOp.getProperties().indices);
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
      derivedVariables.insert(access.getVariable());

      derivativesMap.addDerivedIndices(access.getVariable(),
                                       std::move(derivedIndices));
    }
  }

  return mlir::success();
}

mlir::LogicalResult DerivativesMaterializationPass::collectDerivedIndices(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
    DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
    AlgorithmOp algorithmOp) const {
  return collectDerivedIndicesInAlgorithmRegion(
      modelOp, symbolTableCollection, derivedVariables, derivativesMap,
      mutexCollection, algorithmOp.getBodyRegion());
}

mlir::LogicalResult
DerivativesMaterializationPass::collectDerivedIndicesInAlgorithmRegion(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
    DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
    mlir::Region &region) const {
  llvm::SmallVector<DerOp> derOps;

  region.walk([&](DerOp derOp) { derOps.push_back(derOp); });

  for (DerOp derOp : derOps) {
    llvm::SmallVector<mlir::FlatSymbolRefAttr, 3> symbols;
    llvm::SmallVector<mlir::Value, 3> indices;

    if (mlir::failed(getAccess(derOp.getOperand(), symbols, indices))) {
      derOp.emitOpError() << "Can't obtain the access to the variable";
      return mlir::failure();
    }

    mlir::SymbolRefAttr variable = getSymbolRefFromPath(symbols);
    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(variableShape, modelOp, symbolTableCollection,
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

static std::string getDerivativeName(mlir::SymbolRefAttr variableName) {
  std::string result = "der_" + variableName.getRootReference().str();

  for (mlir::FlatSymbolRefAttr component : variableName.getNestedReferences()) {
    result += "." + component.getValue().str();
  }

  return result;
}

mlir::LogicalResult DerivativesMaterializationPass::createDerivativeVariables(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    DerivativesMap &derivativesMap,
    const llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
    MutexCollection &mutexCollection) {
  mlir::OpBuilder builder(modelOp);
  builder.setInsertionPointToEnd(modelOp.getBody());

  mlir::SymbolTable &symbolTable =
      symbolTableCollection.getSymbolTable(modelOp);

  // Add the new attributes.
  for (mlir::SymbolRefAttr variable : derivedVariables) {
    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(variableShape, modelOp, symbolTableCollection,
                              mutexCollection.symbolTableCollectionMutex,
                              variable))) {
      return mlir::failure();
    }

    auto derVariableOp = builder.create<VariableOp>(
        modelOp.getLoc(), getDerivativeName(variable),
        VariableType::get(variableShape, RealType::get(builder.getContext()),
                          VariabilityProperty::none, IOProperty::none));

    symbolTable.insert(derVariableOp, modelOp.getBody()->end());
    auto derivative = mlir::SymbolRefAttr::get(derVariableOp.getSymNameAttr());
    derivativesMap.setDerivative(variable, derivative);
  }

  return mlir::success();
}

static VariableOp
resolveVariable(ModelOp modelOp,
                mlir::SymbolTableCollection &symbolTableCollection,
                mlir::SymbolRefAttr variable) {
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, variable.getRootReference());

  for (mlir::FlatSymbolRefAttr component : variable.getNestedReferences()) {
    auto recordOp = mlir::cast<RecordOp>(
        mlir::cast<RecordType>(variableOp.getVariableType().unwrap())
            .getRecordOp(symbolTableCollection, moduleOp));

    variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        recordOp, component.getAttr());
  }

  return variableOp;
}

namespace {
class DerOpRemovePattern : public mlir::OpRewritePattern<DerOp> {
public:
  DerOpRemovePattern(mlir::MLIRContext *context,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     std::mutex &symbolTableMutex,
                     const DerivativesMap &derivativesMap)
      : mlir::OpRewritePattern<DerOp>(context),
        symbolTableCollection(&symbolTableCollection),
        symbolTableMutex(&symbolTableMutex), derivativesMap(&derivativesMap) {}

  mlir::LogicalResult
  matchAndRewrite(DerOp op, mlir::PatternRewriter &rewriter) const override {
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

    mlir::Value replacement = rewriter.create<VariableGetOp>(loc, variableOp);

    if (!indices.empty()) {
      replacement = rewriter.create<TensorViewOp>(loc, replacement, indices);
    }

    if (auto tensorType =
            mlir::dyn_cast<mlir::TensorType>(replacement.getType());
        tensorType && tensorType.hasRank()) {
      replacement =
          rewriter.create<TensorExtractOp>(loc, replacement, std::nullopt);
    }

    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

private:
  VariableOp resolveVariable(ModelOp modelOp,
                             mlir::SymbolRefAttr variable) const {
    std::lock_guard<std::mutex> lock(*symbolTableMutex);
    return ::resolveVariable(modelOp, *symbolTableCollection, variable);
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
  std::mutex *symbolTableMutex;
  const DerivativesMap *derivativesMap;
};
} // namespace

mlir::LogicalResult DerivativesMaterializationPass::removeDerOps(
    mlir::SymbolTableCollection &symbolTableCollection,
    const DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
    EquationInstanceOp equationInstanceOp) {
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<DerOpRemovePattern>(&getContext(), symbolTableCollection,
                                   mutexCollection.symbolTableCollectionMutex,
                                   derivativesMap);

  return mlir::applyPatternsGreedily(equationInstanceOp.getTemplate(),
                                     std::move(patterns));
}

mlir::LogicalResult DerivativesMaterializationPass::removeDerOps(
    mlir::SymbolTableCollection &symbolTableCollection,
    const DerivativesMap &derivativesMap, MutexCollection &mutexCollection,
    AlgorithmOp algorithmOp) {
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<DerOpRemovePattern>(&getContext(), symbolTableCollection,
                                   mutexCollection.symbolTableCollectionMutex,
                                   derivativesMap);

  return mlir::applyPatternsGreedily(algorithmOp, std::move(patterns));
}

static mlir::LogicalResult
createStartOp(mlir::OpBuilder &builder,
              mlir::SymbolTableCollection &symbolTableCollection,
              ModelOp modelOp, mlir::SymbolRefAttr variable) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modelOp.getBody());

  VariableOp variableOp =
      resolveVariable(modelOp, symbolTableCollection, variable);

  mlir::Location loc = variableOp.getLoc();

  auto startOp = builder.create<StartOp>(loc, variable, false, false, true);
  assert(startOp.getBodyRegion().empty());
  mlir::Block *bodyBlock = builder.createBlock(&startOp.getBodyRegion());
  builder.setInsertionPointToStart(bodyBlock);

  mlir::Value zero =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  VariableType variableType = variableOp.getVariableType();

  if (!variableType.isScalar()) {
    zero = builder.create<TensorBroadcastOp>(loc, variableType.toTensorType(),
                                             zero);
  }

  builder.create<YieldOp>(loc, zero);
  return mlir::success();
}

static mlir::LogicalResult
createMainEquations(mlir::OpBuilder &builder,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, mlir::SymbolRefAttr derivativeName,
                    const IndexSet &indices) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modelOp.getBody());

  VariableOp variableOp =
      resolveVariable(modelOp, symbolTableCollection, derivativeName);

  mlir::Location loc = variableOp.getLoc();

  auto equationTemplateOp = builder.create<EquationTemplateOp>(loc);

  builder.setInsertionPointToStart(
      equationTemplateOp.createBody(indices.rank()));

  mlir::Value variable = builder.create<VariableGetOp>(loc, variableOp);

  variable = builder.create<TensorExtractOp>(
      loc, variable, equationTemplateOp.getInductionVariables());

  mlir::Value zero =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value lhs = builder.create<EquationSideOp>(loc, variable);
  mlir::Value rhs = builder.create<EquationSideOp>(loc, zero);
  builder.create<EquationSidesOp>(loc, lhs, rhs);

  builder.setInsertionPointAfter(equationTemplateOp);

  auto dynamicOp = builder.create<DynamicOp>(modelOp.getLoc());
  builder.createBlock(&dynamicOp.getBodyRegion());

  auto instanceOp = builder.create<EquationInstanceOp>(loc, equationTemplateOp);

  if (mlir::failed(instanceOp.setIndices(indices, symbolTableCollection))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
DerivativesMaterializationPass::createStartOpsAndDummyEquations(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    const llvm::DenseSet<mlir::SymbolRefAttr> &derivedVariables,
    const DerivativesMap &derivativesMap, MutexCollection &mutexCollection) {
  mlir::OpBuilder builder(modelOp);

  for (mlir::SymbolRefAttr variableName : derivedVariables) {
    auto derivativeName = derivativesMap.getDerivative(variableName);

    if (!derivativeName) {
      continue;
    }

    // Create the start value.
    if (mlir::failed(createStartOp(builder, symbolTableCollection, modelOp,
                                   *derivativeName))) {
      return mlir::failure();
    }

    // Create the equations for the non-derived indices.
    llvm::SmallVector<int64_t, 3> variableDimensions;

    if (mlir::failed(getShape(
            variableDimensions, modelOp, symbolTableCollection,
            mutexCollection.symbolTableCollectionMutex, variableName))) {
      return mlir::failure();
    }

    if (!variableDimensions.empty()) {
      IndexSet nonDerivedIndices = shapeToIndexSet(variableDimensions);

      if (auto derivedIndices =
              derivativesMap.getDerivedIndices(variableName)) {
        nonDerivedIndices -= derivedIndices->get();
      }

      if (!nonDerivedIndices.empty()) {
        if (mlir::failed(createMainEquations(
                builder, symbolTableCollection, modelOp, *derivativeName,
                nonDerivedIndices.getCanonicalRepresentation()))) {
          return mlir::failure();
        }
      }
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createDerivativesMaterializationPass() {
  return std::make_unique<DerivativesMaterializationPass>();
}
} // namespace mlir::bmodelica
