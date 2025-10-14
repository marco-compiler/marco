#include "marco/Dialect/BaseModelica/Transforms/DerivativesMaterialization.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Threading.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_DERIVATIVESMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class DifferentialVariablesSet : public llvm::SetVector<mlir::SymbolRefAttr> {
  mutable std::mutex mutex;

public:
  bool insert(mlir::SymbolRefAttr variable) {
    std::lock_guard lock(mutex);
    return llvm::SetVector<mlir::SymbolRefAttr>::insert(variable);
  }
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

  mlir::LogicalResult
  collectDerivedIndices(ModelOp modelOp,
                        mlir::SymbolTableCollection &symbolTables,
                        DifferentialVariablesSet &newDifferentialVariables,
                        DerivativesMap &derivativesMap,
                        EquationInstanceOp equationInstanceOp) const;

  mlir::LogicalResult collectDerivedIndices(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTables,
      DifferentialVariablesSet &newDifferentialVariables,
      DerivativesMap &derivativesMap, AlgorithmOp algorithmOp) const;

  mlir::LogicalResult collectDerivedIndicesInAlgorithmRegion(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTables,
      DifferentialVariablesSet &newDifferentialVariables,
      DerivativesMap &derivativesMap, mlir::Region &region) const;

  mlir::LogicalResult createDerivativeVariables(
      ModelOp modelOp, mlir::SymbolTableCollection &symbolTables,
      DerivativesMap &derivativesMap,
      const llvm::SetVector<mlir::SymbolRefAttr> &newDifferentialVariables);
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

namespace {
mlir::LogicalResult
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

mlir::SymbolRefAttr
getSymbolRefFromPath(llvm::ArrayRef<mlir::FlatSymbolRefAttr> symbols) {
  assert(!symbols.empty());
  return mlir::SymbolRefAttr::get(symbols[0].getAttr(), symbols.drop_front());
}

VariableOp resolveVariable(ModelOp modelOp,
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

mlir::LogicalResult removeDerOps(mlir::Region &region, ModelOp modelOp,
                                 mlir::SymbolTableCollection &symbolTables,
                                 const DerivativesMap &derivativesMap) {
  mlir::IRRewriter rewriter(region.getContext());
  llvm::SmallVector<DerOp> derOps;
  region.walk([&](DerOp derOp) { derOps.push_back(derOp); });

  for (DerOp derOp : derOps) {
    rewriter.setInsertionPoint(derOp);

    llvm::SmallVector<mlir::FlatSymbolRefAttr, 3> symbols;
    llvm::SmallVector<mlir::Value, 3> indices;

    if (mlir::failed(getAccess(derOp.getOperand(), symbols, indices))) {
      return mlir::failure();
    }

    mlir::SymbolRefAttr variableName = getSymbolRefFromPath(symbols);
    auto derivativeName = derivativesMap.getDerivative(variableName);

    if (!derivativeName) {
      return mlir::failure();
    }

    VariableOp variableOp =
        ::resolveVariable(modelOp, symbolTables, *derivativeName);
    mlir::Value replacement =
        rewriter.create<VariableGetOp>(derOp.getLoc(), variableOp);

    if (!indices.empty()) {
      replacement =
          rewriter.create<TensorViewOp>(derOp.getLoc(), replacement, indices);
    }

    if (auto tensorType =
            mlir::dyn_cast<mlir::TensorType>(replacement.getType());
        tensorType && tensorType.hasRank()) {
      replacement = rewriter.create<TensorExtractOp>(
          derOp.getLoc(), replacement, mlir::ValueRange());
    }

    rewriter.replaceOp(derOp, replacement);
  }

  return mlir::success();
}
} // namespace

mlir::LogicalResult
DerivativesMaterializationPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTables;
  mlir::LockedSymbolTableCollection lockedSymbolTables(symbolTables);

  llvm::SmallVector<EquationInstanceOp> equationInstanceOps;
  llvm::SmallVector<AlgorithmOp> algorithmOps;

  modelOp.collectInitialEquations(equationInstanceOps);
  modelOp.collectMainEquations(equationInstanceOps);

  modelOp.collectInitialAlgorithms(algorithmOps);
  modelOp.collectMainAlgorithms(algorithmOps);

  // Collect the derived indices.
  DifferentialVariablesSet newDifferentialVariables;
  DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  LockedDerivativesMap lockedDerivativesMap(derivativesMap);

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationInstanceOps, [&](EquationInstanceOp equation) {
            return collectDerivedIndices(modelOp, lockedSymbolTables,
                                         newDifferentialVariables,
                                         lockedDerivativesMap, equation);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps, [&](AlgorithmOp algorithmOp) {
            return collectDerivedIndices(modelOp, lockedSymbolTables,
                                         newDifferentialVariables,
                                         lockedDerivativesMap, algorithmOp);
          }))) {
    return mlir::failure();
  }

  // Create the derivative variables.
  if (mlir::failed(createDerivativeVariables(
          modelOp, symbolTables, derivativesMap, newDifferentialVariables))) {
    return mlir::failure();
  }

  // Replace the derivative operations.
  llvm::SmallVector<EquationTemplateOp> equationTemplateOps;

  for (EquationInstanceOp equationInstanceOp : equationInstanceOps) {
    equationTemplateOps.push_back(equationInstanceOp.getTemplate());
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), equationTemplateOps, [&](EquationTemplateOp equation) {
            return ::removeDerOps(equation.getBodyRegion(), modelOp,
                                  lockedSymbolTables, lockedDerivativesMap);
          }))) {
    return mlir::failure();
  }

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), algorithmOps, [&](AlgorithmOp algorithmOp) {
            return ::removeDerOps(algorithmOp.getBodyRegion(), modelOp,
                                  lockedSymbolTables, lockedDerivativesMap);
          }))) {
    return mlir::failure();
  }

  return mlir::success();
}

namespace {
mlir::LogicalResult getShape(llvm::SmallVectorImpl<int64_t> &shape,
                             ModelOp modelOp,
                             mlir::SymbolTableCollection &symbolTables,
                             mlir::SymbolRefAttr variable) {
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto variableOp = symbolTables.lookupSymbolIn<VariableOp>(
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
            .getRecordOp(symbolTables, moduleOp));

    variableOp =
        symbolTables.lookupSymbolIn<VariableOp>(recordOp, component.getAttr());

    if (!variableOp) {
      return mlir::failure();
    }

    auto componentShape = variableOp.getVariableType().getShape();
    shape.append(componentShape.begin(), componentShape.end());
  }

  return mlir::success();
}

IndexSet shapeToIndexSet(llvm::ArrayRef<int64_t> shape) {
  IndexSet result;
  llvm::SmallVector<Range, 3> ranges;

  for (int64_t dimension : shape) {
    ranges.push_back(Range(0, dimension));
  }

  result += MultidimensionalRange(std::move(ranges));
  return result;
}

void collectDerOps(
    llvm::SmallVectorImpl<std::pair<DerOp, EquationPath>> &result,
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

void collectDerOps(
    llvm::SmallVectorImpl<std::pair<DerOp, EquationPath>> &result,
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
  for (size_t i = 0, e = rhsValues.size(); i < e; ++i) {
    collectDerOps(result, rhsValues[i], EquationPath(EquationPath::RIGHT, i));
  }
}
} // namespace

mlir::LogicalResult DerivativesMaterializationPass::collectDerivedIndices(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTables,
    DifferentialVariablesSet &newDifferentialVariables,
    DerivativesMap &derivativesMap,
    EquationInstanceOp equationInstanceOp) const {
  EquationTemplateOp equationTemplateOp = equationInstanceOp.getTemplate();

  llvm::SmallVector<std::pair<DerOp, EquationPath>> derOps;
  collectDerOps(derOps, equationTemplateOp);

  for (const auto &derOp : derOps) {
    llvm::SmallVector<VariableAccess, 1> accesses;

    {
      auto access =
          equationTemplateOp.getAccessAtPath(symbolTables, derOp.second + 0);

      if (!access) {
        return mlir::failure();
      }

      accesses.push_back(std::move(*access));
    }

    for (const VariableAccess &access : accesses) {
      llvm::SmallVector<int64_t, 3> variableShape;

      if (mlir::failed(getShape(variableShape, modelOp, symbolTables,
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

        if (derivedIndices.empty()) {
          derivedIndices = {MultidimensionalRange(extraDimensions)};
        } else {
          derivedIndices = derivedIndices.append(
              IndexSet(MultidimensionalRange(extraDimensions)));
        }
      }

      // Add the derived indices.
      newDifferentialVariables.insert(access.getVariable());

      derivativesMap.addDerivedIndices(access.getVariable(),
                                       std::move(derivedIndices));
    }
  }

  return mlir::success();
}

mlir::LogicalResult DerivativesMaterializationPass::collectDerivedIndices(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    DifferentialVariablesSet &newDifferentialVariables,
    DerivativesMap &derivativesMap, AlgorithmOp algorithmOp) const {
  return collectDerivedIndicesInAlgorithmRegion(
      modelOp, symbolTableCollection, newDifferentialVariables, derivativesMap,
      algorithmOp.getBodyRegion());
}

mlir::LogicalResult
DerivativesMaterializationPass::collectDerivedIndicesInAlgorithmRegion(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    DifferentialVariablesSet &newDifferentialVariables,
    DerivativesMap &derivativesMap, mlir::Region &region) const {
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
                              variable))) {
      return mlir::failure();
    }

    // Add the derived indices.
    newDifferentialVariables.insert(variable);
    derivativesMap.addDerivedIndices(variable, shapeToIndexSet(variableShape));
  }

  return mlir::success();
}

namespace {
std::string getDerivativeName(mlir::SymbolRefAttr variableName) {
  std::string result = "der_" + variableName.getRootReference().str();

  for (mlir::FlatSymbolRefAttr component : variableName.getNestedReferences()) {
    result += "." + component.getValue().str();
  }

  return result;
}
} // namespace

mlir::LogicalResult DerivativesMaterializationPass::createDerivativeVariables(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection,
    DerivativesMap &derivativesMap,
    const llvm::SetVector<mlir::SymbolRefAttr> &newDifferentialVariables) {
  mlir::OpBuilder builder(modelOp);
  builder.setInsertionPointToEnd(modelOp.getBody());

  mlir::SymbolTable &symbolTable =
      symbolTableCollection.getSymbolTable(modelOp);

  // Add the new attributes.
  for (mlir::SymbolRefAttr variable : newDifferentialVariables) {
    // If the derivative map contains the derivative, a variable already exists.
    if (derivativesMap.getDerivative(variable).has_value()) {
      continue;
    }

    llvm::SmallVector<int64_t, 3> variableShape;

    if (mlir::failed(getShape(variableShape, modelOp, symbolTableCollection,
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

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createDerivativesMaterializationPass() {
  return std::make_unique<DerivativesMaterializationPass>();
}
} // namespace mlir::bmodelica
