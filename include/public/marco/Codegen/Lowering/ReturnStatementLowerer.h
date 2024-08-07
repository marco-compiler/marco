#ifndef MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ReturnStatementLowerer : public Lowerer
  {
    public:
      explicit ReturnStatementLowerer(BridgeInterface* bridge);

      [[nodiscard]] bool lower(const ast::ReturnStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H
