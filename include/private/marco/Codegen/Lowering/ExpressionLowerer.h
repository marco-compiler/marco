#ifndef MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ExpressionLowerer : public Lowerer
  {
    public:
      ExpressionLowerer(BridgeInterface* bridge);

      Results lower(const ast::Expression& expression) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H
