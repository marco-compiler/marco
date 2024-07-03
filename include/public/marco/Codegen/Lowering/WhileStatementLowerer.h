#ifndef MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class WhileStatementLowerer : public Lowerer
  {
    public:
      explicit WhileStatementLowerer(BridgeInterface* bridge);

      [[nodiscard]] bool lower(const ast::WhileStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H
