#include "marco/Codegen/Lowering/ComponentReferenceLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ComponentReferenceLowerer::ComponentReferenceLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results ComponentReferenceLowerer::lower(
      const ast::ComponentReference& componentReference)
  {
    mlir::Location location = loc(componentReference.getLocation());

    size_t pathLength = componentReference.getPathLength();
    assert(pathLength >= 1);

    const ast::ComponentReferenceEntry* firstEntry =
        componentReference.getElement(0);

    Reference result = lookupVariable(firstEntry->getName());
    result = lowerSubscripts(result, *firstEntry);

    for (size_t i = 1; i < pathLength; ++i) {
      mlir::Value parent = result.get(location);
      mlir::Type parentType = parent.getType();

      const ast::ComponentReferenceEntry* pathEntry =
          componentReference.getElement(i);

      mlir::Type baseType = parentType;

      if (auto parentArrayType = parentType.dyn_cast<ArrayType>()) {
        baseType = parentArrayType.getElementType();
      }

      if (auto recordType = mlir::dyn_cast<RecordType>(baseType)) {
        auto recordOp = resolveSymbolName<RecordOp>(
            recordType.getName(), getLookupScope());

        auto variableOp = mlir::cast<VariableOp>(
            resolveSymbolName<VariableOp>(pathEntry->getName(), recordOp));

        llvm::SmallVector<int64_t, 3> shape;

        if (auto parentTypeArrayType = parentType.dyn_cast<ArrayType>()) {
          for (int64_t inheritedDimension : parentTypeArrayType.getShape()) {
            shape.push_back(inheritedDimension);
          }
        }

        mlir::Type componentType = variableOp.getVariableType().unwrap();

        if (auto componentArrayType = componentType.dyn_cast<ArrayType>()) {
          for (int64_t componentDimension : componentArrayType.getShape()) {
            shape.push_back(componentDimension);
          }

          componentType = componentArrayType.withShape(shape);
        } else if (!shape.empty()) {
          componentType = ArrayType::get(shape, componentType);
        }

        result = Reference::component(
            builder(), location, parent, componentType, pathEntry->getName());
      }

      result = lowerSubscripts(result, *pathEntry);
    }

    return result;
  }

  Reference ComponentReferenceLowerer::lowerSubscripts(
      Reference current, const ast::ComponentReferenceEntry& entry)
  {
    // Indices in Modelica are 1-based.
    std::vector<mlir::Value> zeroBasedIndices;

    for (size_t i = 0, e = entry.getNumOfSubscripts(); i < e; ++i) {
      mlir::Location indexLoc = loc(entry.getSubscript(i)->getLocation());
      mlir::Value index = lower(*entry.getSubscript(i))[0].get(indexLoc);

      mlir::Value one = builder().create<ConstantOp>(
          index.getLoc(), builder().getIndexAttr(-1));

      mlir::Value zeroBasedIndex = builder().create<AddOp>(
          index.getLoc(), builder().getIndexType(), index, one);

      zeroBasedIndices.push_back(zeroBasedIndex);
    }

    if (!zeroBasedIndices.empty()) {
      mlir::Location location = loc(entry.getLocation());
      mlir::Value array = current.get(loc(entry.getLocation()));

      mlir::Value result = builder().create<SubscriptionOp>(
          location, array, zeroBasedIndices);

      return Reference::memory(builder(), result);
    }

    return current;
  }
}
