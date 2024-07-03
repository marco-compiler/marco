#ifndef MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H
#define MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ArrayGeneratorLowerer : public Lowerer
  {
    public:
      explicit ArrayGeneratorLowerer(BridgeInterface* bridge);

      std::optional<Results> lower(const ast::ArrayGenerator& array) override;

    protected:
      using Lowerer::lower;

    private:
      std::optional<Results> lower(const ast::ArrayConstant& array);

      std::optional<Results> lower(const ast::ArrayForGenerator& array);

      void computeShape(const ast::ArrayGenerator& array, llvm::SmallVectorImpl<int64_t>& outShape);

      [[nodiscard]] bool 
      lowerValues(const ast::Expression& array, llvm::SmallVectorImpl<mlir::Value>& outValues);
  };
}

#endif // MARCO_CODEGEN_LOWERING_ARRAYGENERATORLOWERER_H
