#include "marco/Codegen/Lowering/EquationSectionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  EquationSectionLowerer::EquationSectionLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void EquationSectionLowerer::lower(
      const ast::EquationSection& equationSection)
  {
    if (equationSection.isInitial()) {
      auto initialModelOp = builder().create<InitialModelOp>(
          loc(equationSection.getLocation()));

      mlir::Block* bodyBlock =
          builder().createBlock(&initialModelOp.getBodyRegion());

      builder().setInsertionPointToStart(bodyBlock);

      for (size_t i = 0, e = equationSection.getNumOfEquations(); i < e; ++i) {
        lower(*equationSection.getEquation(i));
      }

      builder().setInsertionPointAfter(initialModelOp);
    } else {
      auto mainModelOp = builder().create<MainModelOp>(
          loc(equationSection.getLocation()));

      mlir::Block* bodyBlock =
          builder().createBlock(&mainModelOp.getBodyRegion());

      builder().setInsertionPointToStart(bodyBlock);

      for (size_t i = 0, e = equationSection.getNumOfEquations(); i < e; ++i) {
        lower(*equationSection.getEquation(i));
      }

      builder().setInsertionPointAfter(mainModelOp);
    }
  }
}
