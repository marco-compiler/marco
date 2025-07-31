#ifndef MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class StandardFunctionLowerer : public Lowerer {
public:
  explicit StandardFunctionLowerer(BridgeInterface *bridge);

  void declare(const ast::StandardFunction &function) override;

  [[nodiscard]] bool
  declareVariables(const ast::StandardFunction &function) override;

  [[nodiscard]] bool lower(const ast::StandardFunction &function) override;


protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;

  std::optional<mlir::bmodelica::VariableType>
  getVariableType(const ast::VariableType &type,
                  const ast::TypePrefix &typePrefix);

private:
  [[nodiscard]] bool lowerVariableDefaultValue(const ast::Member &variable);

  bool isRecordConstructor(const ast::StandardFunction &function);
  mlir::Type lowerArg(const ast::Expression &expression); 

};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
