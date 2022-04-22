#ifndef MARCO_CODEGEN_LOWERING_MODELBRIDGE_H
#define MARCO_CODEGEN_LOWERING_MODELBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ModelLowerer : public Lowerer
  {
    public:
      ModelLowerer(LoweringContext* context, BridgeInterface* bridge);

      std::vector<mlir::Operation*> lower(const ast::Model& model);

    protected:
      using Lowerer::lower;

    private:
      void lower(const ast::Member& member);

      void createMemberTrivialEquation(
          mlir::modelica::ModelOp modelOp, const ast::Member& member, const ast::Expression& expression);
  };
}

#endif // MARCO_CODEGEN_LOWERING_MODELBRIDGE_H
