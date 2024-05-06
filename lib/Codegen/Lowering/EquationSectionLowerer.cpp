#include "marco/Codegen/Lowering/EquationSectionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

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
      auto initialOp = builder().create<InitialOp>(
          loc(equationSection.getLocation()));

      mlir::Block* bodyBlock =
          builder().createBlock(&initialOp.getBodyRegion());

      builder().setInsertionPointToStart(bodyBlock);

      for (size_t i = 0, e = equationSection.getNumOfEquations(); i < e; ++i) {
        lower(*equationSection.getEquation(i));
      }

      builder().setInsertionPointAfter(initialOp);
    } else {
      auto dynamicOp = builder().create<DynamicOp>(
          loc(equationSection.getLocation()));

      mlir::Block* bodyBlock =
          builder().createBlock(&dynamicOp.getBodyRegion());

      builder().setInsertionPointToStart(bodyBlock);

      for (size_t i = 0, e = equationSection.getNumOfEquations(); i < e; ++i) {
        lower(*equationSection.getEquation(i));
      }

      builder().setInsertionPointAfter(dynamicOp);
    }
  }
}
