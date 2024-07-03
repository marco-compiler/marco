#ifndef MARCO_CODEGEN_LOWERING_WHENSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_WHENSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class WhenStatementLowerer : public Lowerer
  {
    public:
      explicit WhenStatementLowerer(BridgeInterface* bridge);

      [[nodiscard]] bool lower(const ast::WhenStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_WHENSTATEMENTLOWERER_H
