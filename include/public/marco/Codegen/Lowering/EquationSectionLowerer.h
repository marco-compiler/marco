#ifndef MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class EquationSectionLowerer : public Lowerer
  {
    public:
      explicit EquationSectionLowerer(BridgeInterface* bridge);

      __attribute__((warn_unused_result)) bool 
          lower(const ast::EquationSection& equationSection) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
