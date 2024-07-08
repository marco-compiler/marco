#include "marco/Codegen/Lowering/ComponentReferenceLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  ComponentReferenceLowerer::ComponentReferenceLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  std::optional<Results> ComponentReferenceLowerer::lower(
      const ast::ComponentReference& componentReference)
  {
    mlir::Location location = loc(componentReference.getLocation());

    size_t pathLength = componentReference.getPathLength();
    assert(pathLength >= 1);

    const ast::ComponentReferenceEntry* firstEntry =
        componentReference.getElement(0);

    std::optional<Reference> result = lookupVariable(firstEntry->getName());
    if (!result) {
      emitIdentifierError(IdentifierError::IdentifierType::VARIABLE, firstEntry->getName(), 
                          getVariablesSymbolTable().getVariables(true), 
                          firstEntry->getLocation());
      return std::nullopt;
    }

    result = lowerSubscripts(*result, *firstEntry);
    if (!result) {
      return std::nullopt;
    }

    for (size_t i = 1; i < pathLength; ++i) {
      mlir::Value parent = result->get(location);
      mlir::Type parentType = parent.getType();

      const ast::ComponentReferenceEntry* pathEntry =
          componentReference.getElement(i);

      mlir::Type baseType = parentType;

      if (auto parentShapedType = parentType.dyn_cast<mlir::ShapedType>()) {
        baseType = parentShapedType.getElementType();
      }

      if (auto recordType = mlir::dyn_cast<RecordType>(baseType)) {
        auto recordOp = resolveTypeFromRoot(recordType.getName());
        assert(recordOp != nullptr);

        mlir::Operation *op = resolveSymbolName<VariableOp>(pathEntry->getName(), recordOp);
        if (op == nullptr) {
          std::set<std::string> visibleFields;
          getVisibleSymbols<VariableOp>(recordOp, visibleFields);

          emitIdentifierError(IdentifierError::IdentifierType::FIELD, pathEntry->getName(), 
                              visibleFields, pathEntry->getLocation());
          return std::nullopt;
        }

        auto variableOp = mlir::cast<VariableOp>(op);

        llvm::SmallVector<int64_t, 3> shape;

        if (auto parentShapedType = parentType.dyn_cast<mlir::ShapedType>()) {
          for (int64_t inheritedDimension : parentShapedType.getShape()) {
            shape.push_back(inheritedDimension);
          }
        }

        mlir::Type componentType = variableOp.getVariableType().unwrap();

        if (auto componentShapedType = componentType.dyn_cast<mlir::ShapedType>()) {
          for (int64_t componentDimension : componentShapedType.getShape()) {
            shape.push_back(componentDimension);
          }

          componentType = componentShapedType.clone(shape).cast<mlir::Type>();
        } else if (!shape.empty()) {
          componentType = mlir::RankedTensorType::get(shape, componentType);
        }

        result = Reference::component(
            builder(), location, parent, componentType, pathEntry->getName());
      }

      result = lowerSubscripts(*result, *pathEntry);
      if (!result) {
        return std::nullopt;
      }
    }

    return result;
  }

  std::optional<Reference> ComponentReferenceLowerer::lowerSubscripts(
      Reference current, const ast::ComponentReferenceEntry& entry)
  {
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

      mlir::Location location = loc(entry.getLocation());
      mlir::Value tensor = current.get(loc(entry.getLocation()));
      int64_t sourceRank = tensor.getType().cast<mlir::TensorType>().getRank();

      if (sourceRank > static_cast<int64_t>(subscripts.size())) {
        mlir::Value unboundedRange =
            builder().create<UnboundedRangeOp>(location);

        fullRankSubscripts.append(
            sourceRank - static_cast<int64_t>(subscripts.size()),
            unboundedRange);
      }

      fullRankSubscripts.append(subscripts);

      mlir::Value result = builder().create<TensorViewOp>(
          location, tensor, fullRankSubscripts);

      return Reference::tensor(builder(), result);
    }

    return current;
  }
}
