#ifndef MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class StatementLowerer : public Lowerer
  {
    public:
      explicit StatementLowerer(BridgeInterface* bridge);

      __attribute__((warn_unused_result)) bool lower(const ast::Statement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
