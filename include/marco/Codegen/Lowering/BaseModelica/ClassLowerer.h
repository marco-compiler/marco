#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ClassLowerer : public Lowerer {
public:
  explicit ClassLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::Class &cls) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::Class &cls) override;

  [[nodiscard]] bool declare(const ast::bmodelica::Member &variable) override;

  [[nodiscard]] bool lower(const ast::bmodelica::Class &cls) override;

  [[nodiscard]] bool lowerClassBody(const ast::bmodelica::Class &cls) override;

  [[nodiscard]] bool
  createBindingEquation(const ast::bmodelica::Member &variable,
                        const ast::bmodelica::Expression &expression) override;

  [[nodiscard]] bool
  lowerStartAttribute(mlir::SymbolRefAttr variable,
                      const ast::bmodelica::Expression &expression, bool fixed,
                      bool each) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;

  std::optional<mlir::bmodelica::VariableType>
  getVariableType(const ast::bmodelica::VariableType &type,
                  const ast::bmodelica::TypePrefix &typePrefix);

  [[nodiscard]] bool
  lowerVariableDimensionConstraints(mlir::SymbolTable &symbolTable,
                                    const ast::bmodelica::Member &variable);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSLOWERER_H
