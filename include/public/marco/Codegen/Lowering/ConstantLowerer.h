#ifndef MARCO_CODEGEN_LOWERING_CONSTANTLOWERER_H
#define MARCO_CODEGEN_LOWERING_CONSTANTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include <string>

namespace marco::codegen::lowering
{
  class ConstantLowerer : public Lowerer
  {
    public:
      explicit ConstantLowerer(BridgeInterface* bridge);

      std::optional<Results> lower(const ast::Constant& constant) override;

      mlir::TypedAttr operator()(bool value);
      mlir::TypedAttr operator()(int64_t value);
      mlir::TypedAttr operator()(double value);
      mlir::TypedAttr operator()(std::string value);
  };
}

#endif // MARCO_CODEGEN_LOWERING_CONSTANTLOWERER_H
