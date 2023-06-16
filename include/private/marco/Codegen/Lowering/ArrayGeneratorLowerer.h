#ifndef MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H
#define MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ArrayGeneratorLowerer : public Lowerer
  {
    public:
      ArrayGeneratorLowerer(BridgeInterface* bridge);

      Results lower(const ast::ArrayGenerator& array) override;

    protected:
      using Lowerer::lower;

    private:
      Results lower(const ast::ArrayConstant& array);

      Results lower(const ast::ArrayForGenerator& array);
  };
}

#endif // MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H
