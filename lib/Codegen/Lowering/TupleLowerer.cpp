#include "marco/Codegen/Lowering/BaseModelica/TupleLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
TupleLowerer::TupleLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

std::optional<Results> TupleLowerer::lower(const ast::bmodelica::Tuple &tuple) {
  Results result;

  for (size_t i = 0, e = tuple.size(); i < e; ++i) {
    auto values = lower(*tuple.getExpression(i));
    if (!values) {
      return std::nullopt;
    }

    // The only way to have multiple returns is to call a function, but
    // this is forbidden in a tuple declaration. In fact, a tuple is just
    // a container of references.
    assert(values->size() == 1);
    result.append((*values)[0]);
  }

  return result;
}
} // namespace marco::codegen::lowering::bmodelica
