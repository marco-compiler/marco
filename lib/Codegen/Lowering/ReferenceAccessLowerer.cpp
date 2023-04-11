#include "marco/Codegen/Lowering/ReferenceAccessLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ReferenceAccessLowerer::ReferenceAccessLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results ReferenceAccessLowerer::lower(
      const ast::ReferenceAccess& referenceAccess)
  {
    mlir::Location location = loc(referenceAccess.getLocation());

    llvm::ArrayRef<std::string> pathVariables =
        referenceAccess.getPathVariables();

    Reference result = lookupVariable(pathVariables[0]);

    if (pathVariables.size() > 1) {
      for (size_t i = 1, e = pathVariables.size(); i < e; ++i) {
        mlir::Value parent = result.get(location);
        mlir::Type variableType = parent.getType();

        if (auto recordType = mlir::dyn_cast<RecordType>(variableType)) {
          auto recordOp = resolveSymbolName<RecordOp>(
              recordType.getName(), getLookupScope());

          auto variableOp = mlir::cast<VariableOp>(
              resolveSymbolName<VariableOp>(pathVariables[i], recordOp));

          result = Reference::component(
              builder(), location, parent,
              variableOp.getVariableType().unwrap(),
              pathVariables[i]);
        }
      }
    }

    return result;
  }
}
