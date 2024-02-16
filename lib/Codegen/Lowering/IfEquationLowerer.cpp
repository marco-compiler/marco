#include "marco/Codegen/Lowering/IfEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  IfEquationLowerer::IfEquationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void IfEquationLowerer::lower(const ast::IfEquation& equation)
  {
    llvm_unreachable("If equation is not implemented yet");
  }
}
