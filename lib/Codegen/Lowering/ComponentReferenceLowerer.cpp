#include "marco/Codegen/Lowering/BaseModelica/ComponentReferenceLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
ComponentReferenceLowerer::ComponentReferenceLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

std::optional<Results> ComponentReferenceLowerer::lower(
    const ast::bmodelica::ComponentReference &componentReference) {
  mlir::Location location = loc(componentReference.getLocation());

  size_t pathLength = componentReference.getPathLength();
  assert(pathLength >= 1);

  const ast::bmodelica::ComponentReferenceEntry *firstEntry =
      componentReference.getElement(0);

  std::optional<Reference> result = lookupVariable(firstEntry->getName());

  if (!result) {
    emitIdentifierError(IdentifierError::IdentifierType::VARIABLE,
                        firstEntry->getName(),
                        getVariablesSymbolTable().getVariables(true),
                        firstEntry->getLocation());
    return std::nullopt;
  }

  result = lowerSubscripts(*result, *firstEntry, true, pathLength == 1);

  if (!result) {
    return std::nullopt;
  }

  for (size_t i = 1; i < pathLength; ++i) {
    mlir::Value parent = result->get(location);
    mlir::Type parentType = parent.getType();

    const ast::bmodelica::ComponentReferenceEntry *pathEntry =
        componentReference.getElement(i);

    mlir::Type baseType = parentType;

    if (auto parentShapedType = mlir::dyn_cast<mlir::ShapedType>(parentType)) {
      baseType = parentShapedType.getElementType();
    }

    if (auto recordType = mlir::dyn_cast<RecordType>(baseType)) {
      auto recordOp = resolveTypeFromRoot(recordType.getName());
      assert(recordOp != nullptr);

      mlir::Operation *op =
          resolveSymbolName<VariableOp>(pathEntry->getName(), recordOp);

      if (!op) {
        std::set<std::string> visibleFields;
        getVisibleSymbols<VariableOp>(recordOp, visibleFields);

        emitIdentifierError(IdentifierError::IdentifierType::FIELD,
                            pathEntry->getName(), visibleFields,
                            pathEntry->getLocation());
        return std::nullopt;
      }

      auto variableOp = mlir::cast<VariableOp>(op);

      llvm::SmallVector<int64_t, 3> shape;

      if (auto parentShapedType =
              mlir::dyn_cast<mlir::ShapedType>(parentType)) {
        llvm::append_range(shape, parentShapedType.getShape());
      }

      mlir::Type componentType = variableOp.getVariableType().unwrap();

      if (auto componentShapedType =
              mlir::dyn_cast<mlir::ShapedType>(componentType)) {
        llvm::append_range(shape, componentShapedType.getShape());
        componentType =
            mlir::cast<mlir::Type>(componentShapedType.clone(shape));
      } else if (!shape.empty()) {
        componentType = mlir::RankedTensorType::get(shape, componentType);
      }

      result = Reference::component(builder(), location, parent, componentType,
                                    pathEntry->getName());
    }

    result = lowerSubscripts(*result, *pathEntry, false, i == pathLength - 1);

    if (!result) {
      return std::nullopt;
    }
  }

  return result;
}

std::optional<Reference> ComponentReferenceLowerer::lowerSubscripts(
    Reference current, const ast::bmodelica::ComponentReferenceEntry &entry,
    bool isFirst, bool isLast) {
  llvm::SmallVector<mlir::Value> subscripts;

  for (size_t i = 0, e = entry.getNumOfSubscripts(); i < e; ++i) {
    std::optional<Results> loweredSubscript = lower(*entry.getSubscript(i));

    if (!loweredSubscript) {
      return std::nullopt;
    }

    Result &index = (*loweredSubscript)[0];
    mlir::Value subscript = index.get(index.getLoc());
    subscripts.push_back(subscript);
  }

  if (!subscripts.empty()) {
    llvm::SmallVector<mlir::Value> fullRankSubscripts;

    // isFirst \ isLast | true              | false
    // true             | Lowered           | Lowered + unbound
    // false            | Unbound + lowered | Lowered

    if (isFirst) {
      fullRankSubscripts.append(subscripts);
    }

    mlir::Location location = loc(entry.getLocation());
    mlir::Value tensor = current.get(loc(entry.getLocation()));

    if ((!isFirst && isLast) || (isFirst && !isLast)) {
      int64_t sourceRank =
          mlir::cast<mlir::TensorType>(tensor.getType()).getRank();
      auto providedSubscripts = static_cast<int64_t>(subscripts.size());

      if (sourceRank > providedSubscripts) {
        mlir::Value unboundedRange =
            builder().create<UnboundedRangeOp>(location);

        fullRankSubscripts.append(sourceRank - providedSubscripts,
                                  unboundedRange);
      }
    }

    if (!isFirst) {
      fullRankSubscripts.append(subscripts);
    }

    mlir::Value result =
        builder().create<TensorViewOp>(location, tensor, fullRankSubscripts);

    return Reference::tensor(builder(), result);
  }

  return current;
}
} // namespace marco::codegen::lowering::bmodelica
