#ifndef MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class BreakStatementLowerer : public Lowerer
  {
    public:
      explicit BreakStatementLowerer(BridgeInterface* bridge);

      [[nodiscard]] bool lower(const ast::BreakStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H
