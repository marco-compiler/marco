#include "marco/Dialect/BaseModelica/Transforms/DerivativesInitialization.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_DERIVATIVESINITIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class DerivativesInitializationPass final
    : public impl::DerivativesInitializationPassBase<
          DerivativesInitializationPass> {
  using DerivativesInitializationPassBase::DerivativesInitializationPassBase;

public:
  void runOnOperation() override;

private:
  static mlir::LogicalResult processModelOp(ModelOp modelOp);
};

mlir::LogicalResult getShape(llvm::SmallVectorImpl<int64_t> &shape,
                             ModelOp modelOp,
                             mlir::SymbolTableCollection &symbolTableCollection,
                             mlir::SymbolRefAttr variable) {
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

IndexSet shapeToIndexSet(const llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<Range> ranges = llvm::map_to_vector(
      shape, [](const int64_t dimension) { return Range(0, dimension); });

  IndexSet result;
  result += MultidimensionalRange(std::move(ranges));
  return result;
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

mlir::LogicalResult
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

mlir::LogicalResult
createMainEquations(mlir::OpBuilder &builder,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, mlir::SymbolRefAttr derivativeName,
                    llvm::ArrayRef<MultidimensionalRange> indices) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modelOp.getBody());

  VariableOp variableOp =
      resolveVariable(modelOp, symbolTableCollection, derivativeName);

  mlir::Location loc = variableOp.getLoc();

  auto equationTemplateOp = builder.create<EquationTemplateOp>(loc);

  builder.setInsertionPointToStart(equationTemplateOp.createBody(
      indices.empty() ? 0 : indices.front().rank()));

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
DerivativesInitializationPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;
  DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  // Create the start values for all the indices and the equations for the
  // indices of array variables that are not derived.
  mlir::OpBuilder builder(modelOp);
  for (mlir::SymbolRefAttr variableName :
       derivativesMap.getDerivedVariables()) {
    auto derivativeName = derivativesMap.getDerivative(variableName);
    assert(derivativeName && "derivative not found");

    // Create the start value.
    if (mlir::failed(createStartOp(builder, symbolTableCollection, modelOp,
                                   *derivativeName))) {
      return mlir::failure();
    }

    // Create the equations for the non-derived indices.
    llvm::SmallVector<int64_t, 3> variableDimensions;

    // Get the shape of the variable original.
    if (mlir::failed(getShape(variableDimensions, modelOp,
                              symbolTableCollection, variableName))) {
      return mlir::failure();
    }

    // If the variable is not a scalar, see which indices are derived.
    if (!variableDimensions.empty()) {
      IndexSet nonDerivedIndices = shapeToIndexSet(variableDimensions);

      if (auto derivedIndices =
              derivativesMap.getDerivedIndices(variableName)) {
        nonDerivedIndices -= derivedIndices->get();
      }

      if (!nonDerivedIndices.empty()) {
        llvm::SmallVector<MultidimensionalRange> compactRanges;
        nonDerivedIndices.getCompactRanges(compactRanges);

        if (mlir::failed(createMainEquations(builder, symbolTableCollection,
                                             modelOp, *derivativeName,
                                             compactRanges))) {
          return mlir::failure();
        }
      }
    }
  }

  return mlir::success();
}
} // namespace

void DerivativesInitializationPass::runOnOperation() {
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

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createDerivativesInitializationPass() {
  return std::make_unique<DerivativesInitializationPass>();
}
} // namespace mlir::bmodelica
