#ifndef MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class StandardFunctionLowerer : public Lowerer
  {
    public:
      StandardFunctionLowerer(BridgeInterface* bridge);

      void declare(const ast::StandardFunction& function) override;

      void lower(const ast::StandardFunction& function) override;

    protected:
      using Lowerer::declare;
      using Lowerer::lower;

    private:
      void lowerVariableDefaultValue(const ast::Member& variable);
  };
}

#endif // MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
