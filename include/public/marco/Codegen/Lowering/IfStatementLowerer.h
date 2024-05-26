#ifndef MARCO_CODEGEN_LOWERING_IFSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_IFSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class IfStatementLowerer : public Lowerer
  {
    public:
      explicit IfStatementLowerer(BridgeInterface* bridge);

      void lower(const ast::IfStatement& statement) override;

    protected:
      using Lowerer::lower;

    private:
      mlir::Value lowerCondition(const ast::Expression& expression);

      void lower(const ast::StatementsBlock& statementsBlock);
  };
}

#endif // MARCO_CODEGEN_LOWERING_IFSTATEMENTLOWERER_H
