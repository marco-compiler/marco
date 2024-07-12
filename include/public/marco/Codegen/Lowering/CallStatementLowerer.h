#ifndef MARCO_CODGEN_LOWERING_CALLSTATEMENTLOWERER_H
#define MARCO_CODGEN_LOWERING_CALLSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class CallStatementLowerer : public Lowerer
  {
    public:
      explicit CallStatementLowerer(BridgeInterface *bridge);

      [[nodiscard]] bool lower(const ast::CallStatement &statement) override;

    protected:
      using Lowerer::lower;

  };
} // namespace marco::codegen::lowering


#endif /* ifndef MARCO_CODGEN_LOWERING_CALLSTATEMENTLOWERER_H */
