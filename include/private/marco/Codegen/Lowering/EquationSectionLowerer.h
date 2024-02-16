#ifndef MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class EquationSectionLowerer : public Lowerer
  {
    public:
      EquationSectionLowerer(BridgeInterface* bridge);

      void lower(const ast::EquationSection& equationSection) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
