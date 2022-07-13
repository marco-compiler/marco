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

      void createMemberEquation(const ast::Member& member, const ast::Expression& expression);

      void lowerStartAttribute(const ast::Member& member, const ast::Expression& expression, bool fixed, bool each);
  };
}

#endif // MARCO_CODEGEN_LOWERING_MODELBRIDGE_H
