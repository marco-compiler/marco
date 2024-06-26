#ifndef MARCO_CODEGEN_BRIDGEINTERFACE_H
#define MARCO_CODEGEN_BRIDGEINTERFACE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/Results.h"

namespace marco::codegen::lowering
{
  class BridgeInterface
  {
    public:
      virtual ~BridgeInterface();

      virtual LoweringContext& getContext() = 0;

      virtual const LoweringContext& getContext() const = 0;

      virtual mlir::Operation* getRoot() const = 0;

      virtual void declare(const ast::Class& cls) = 0;

      virtual void declare(const ast::Model& model) = 0;

      virtual void declare(const ast::Package& package) = 0;

      virtual void declare(const ast::PartialDerFunction& function) = 0;

      virtual void declare(const ast::Record& record) = 0;

      virtual void declare(const ast::StandardFunction& function) = 0;

      virtual void declareVariables(const ast::Class& cls) = 0;

      virtual void declareVariables(const ast::Model& model) = 0;

      virtual void declareVariables(const ast::Package& package) = 0;

      virtual void declareVariables(const ast::PartialDerFunction& function) = 0;

      virtual void declareVariables(const ast::Record& record) = 0;

      virtual void declareVariables(const ast::StandardFunction& function) = 0;

      virtual void declare(const ast::Member& variable) = 0;

      virtual void lower(const ast::Class& cls) = 0;

      virtual void lower(const ast::Model& model) = 0;

      virtual void lower(const ast::Package& package) = 0;

      virtual void lower(const ast::PartialDerFunction& function) = 0;

      virtual void lower(const ast::Record& record) = 0;

      virtual void lower(const ast::StandardFunction& function) = 0;

      virtual void lowerClassBody(const ast::Class& cls) = 0;

      virtual void createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) = 0;

      virtual void lowerStartAttribute(
          mlir::SymbolRefAttr variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) = 0;

      virtual Results lower(const ast::Expression& expression) = 0;

      virtual Results lower(const ast::ArrayGenerator& array) = 0;

      virtual Results lower(const ast::Call& call) = 0;

      virtual Results lower(const ast::Constant& constant) = 0;

      virtual Results lower(const ast::Operation& operation) = 0;

      virtual Results lower(
          const ast::ComponentReference& componentReference) = 0;

      virtual Results lower(const ast::Tuple& tuple) = 0;

      virtual Results lower(const ast::Subscript& subscript) = 0;

      virtual void lower(const ast::EquationSection& node) = 0;

      virtual void lower(const ast::Equation& equation) = 0;

      virtual void lower(const ast::EqualityEquation& equation) = 0;

      virtual void lower(const ast::ForEquation& equation) = 0;

      virtual void lower(const ast::IfEquation& equation) = 0;

      virtual void lower(const ast::WhenEquation& equation) = 0;

      virtual void lower(const ast::Algorithm& algorithm) = 0;

      virtual void lower(const ast::Statement& statement) = 0;

      virtual void lower(const ast::AssignmentStatement& statement) = 0;

      virtual void lower(const ast::BreakStatement& statement) = 0;

      virtual void lower(const ast::ForStatement& statement) = 0;

      virtual void lower(const ast::IfStatement& statement) = 0;

      virtual void lower(const ast::ReturnStatement& statement) = 0;

      virtual void lower(const ast::WhenStatement& statement) = 0;

      virtual void lower(const ast::WhileStatement& statement) = 0;
  };
}

#endif // MARCO_CODEGEN_BRIDGEINTERFACE_H
