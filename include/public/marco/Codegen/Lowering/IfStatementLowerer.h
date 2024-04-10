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

      __attribute__((warn_unused_result)) bool lower(const ast::IfStatement& statement) override;

    protected:
      using Lowerer::lower;

    private:
      std::optional<mlir::Value> lowerCondition(const ast::Expression& expression);

      __attribute__((warn_unused_result)) bool lower(const ast::StatementsBlock& statementsBlock);
  };
}

#endif // MARCO_CODEGEN_LOWERING_IFSTATEMENTLOWERER_H
