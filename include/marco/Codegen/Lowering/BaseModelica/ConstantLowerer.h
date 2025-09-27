#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CONSTANTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CONSTANTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"
#include <string>

namespace marco::codegen::lowering::bmodelica {
class ConstantLowerer : public Lowerer {
public:
  explicit ConstantLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::Constant &constant) override;

  mlir::TypedAttr operator()(bool value);
  mlir::TypedAttr operator()(int64_t value);
  mlir::TypedAttr operator()(double value);
  mlir::TypedAttr operator()(std::string value);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CONSTANTLOWERER_H
