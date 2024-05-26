#ifndef MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class AssignmentStatementLowerer : public Lowerer
  {
    public:
      explicit AssignmentStatementLowerer(BridgeInterface* bridge);

      void lower(const ast::AssignmentStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H
