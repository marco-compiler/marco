#ifndef MARCO_CODEGEN_LOWERING_CLASSLOWERER_H
#define MARCO_CODEGEN_LOWERING_CLASSLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ClassLowerer : public Lowerer
  {
    public:
      explicit ClassLowerer(BridgeInterface* bridge);

      void declare(const ast::Class& cls) override;

      [[nodiscard]] bool declareVariables(const ast::Class& cls) override;

      [[nodiscard]] bool declare(const ast::Member& variable) override;

      [[nodiscard]] bool lower(const ast::Class& cls) override;

      [[nodiscard]] bool lowerClassBody(const ast::Class& cls) override;

      [[nodiscard]] bool createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) override;

      [[nodiscard]] bool lowerStartAttribute(
          mlir::SymbolRefAttr variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) override;

    protected:
      using Lowerer::declare;
      using Lowerer::declareVariables;
      using Lowerer::lower;

      std::optional<mlir::bmodelica::VariableType> getVariableType(
          const ast::VariableType& type,
          const ast::TypePrefix& typePrefix);

      [[nodiscard]] bool lowerVariableDimensionConstraints(
          mlir::SymbolTable& symbolTable,
          const ast::Member& variable);
  };
}

#endif // MARCO_CODEGEN_LOWERING_CLASSLOWERER_H
