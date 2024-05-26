#ifndef MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ForStatementLowerer : public Lowerer
  {
    public:
      explicit ForStatementLowerer(BridgeInterface* bridge);

      void lower(const ast::ForStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H
