#include "marco/Codegen/Lowering/TupleLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  TupleLowerer::TupleLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results TupleLowerer::lower(const ast::Tuple& tuple)
  {
    Results result;

    for (size_t i = 0, e = tuple.size(); i < e; ++i) {
      auto values = lower(*tuple.getExpression(i));

      // The only way to have multiple returns is to call a function, but
      // this is forbidden in a tuple declaration. In fact, a tuple is just
      // a container of references.
      assert(values.size() == 1);
      result.append(values[0]);
    }

    return result;
  }
}
