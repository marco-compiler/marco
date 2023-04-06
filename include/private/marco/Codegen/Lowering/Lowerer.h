#ifndef MARCO_CODEGEN_LOWERING_LOWERER_H
#define MARCO_CODEGEN_LOWERING_LOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering
{
  class Lowerer : public BridgeInterface
  {
    public:
      using VariablesScope = LoweringContext::VariablesScope;
      using LookupScopeGuard = LoweringContext::LookupScopeGuard;

      Lowerer(BridgeInterface* bridge);

      virtual ~Lowerer();

    protected:
      /// Helper to convert an AST location to a MLIR location.
      mlir::Location loc(const SourcePosition& location);

      /// Helper to convert an AST location range to a MLIR location.
      mlir::Location loc(const SourceRange& location);

      /// @name Utility getters.
      /// {

      mlir::OpBuilder& builder();

      mlir::SymbolTableCollection& getSymbolTable();

      LoweringContext::VariablesSymbolTable& getVariablesSymbolTable();

      mlir::Operation* getLookupScope();

      void pushLookupScope(mlir::Operation* lookupScope);

      mlir::Operation* getClass(const ast::Class& cls);

      mlir::Operation* resolveClassName(
          llvm::StringRef name,
          mlir::Operation* currentScope);

      template<typename... T>
      mlir::Operation* resolveSymbolName(
          llvm::StringRef name,
          mlir::Operation* currentScope)
      {
        return resolveSymbolName(name, currentScope, [](mlir::Operation* op) {
          return mlir::isa<T...>(op);
        });
      }

      Reference lookupVariable(llvm::StringRef name);

      void insertVariable(llvm::StringRef name, Reference reference);

      mlir::Type getMostGenericScalarType(mlir::Type first, mlir::Type second);

      bool isScalarType(mlir::Type type);

      /// }
      /// @name Forwarded methods.
      /// {

      LoweringContext& getContext() override;

      const LoweringContext& getContext() const override;

      mlir::Operation* getRoot() const override;

      virtual void declare(const ast::Class& node) override;

      virtual void declare(const ast::Model& node) override;

      virtual void declare(const ast::Package& node) override;

      virtual void declare(const ast::PartialDerFunction& node) override;

      virtual void declare(const ast::Record& node) override;

      virtual void declare(const ast::StandardFunction& node) override;

      virtual void declareVariables(const ast::Class& node) override;

      virtual void declareVariables(const ast::Model& model) override;

      virtual void declareVariables(const ast::Package& package) override;

      virtual void declareVariables(
          const ast::PartialDerFunction& function) override;

      virtual void declareVariables(const ast::Record& record) override;

      virtual void declareVariables(
          const ast::StandardFunction& function) override;

      virtual void declare(const ast::Member& node) override;

      virtual void lower(const ast::Class& node) override;

      virtual void lower(const ast::Model& node) override;

      virtual void lower(const ast::Package& node) override;

      virtual void lower(const ast::PartialDerFunction& node) override;

      virtual void lower(const ast::Record& node) override;

      virtual void lower(const ast::StandardFunction& node) override;

      virtual void lowerClassBody(const ast::Class& node) override;

      virtual void createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) override;

      virtual void lowerStartAttribute(
          const ast::Member& variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) override;

      virtual Results lower(const ast::Expression& expression) override;

      virtual Results lower(const ast::Array& node) override;

      virtual Results lower(const ast::Call& node) override;

      virtual Results lower(const ast::Constant& constant) override;

      virtual Results lower(const ast::Operation& operation) override;

      virtual Results lower(
          const ast::ReferenceAccess& referenceAccess) override;

      virtual Results lower(const ast::Tuple& tuple) override;

      virtual void lower(const ast::Algorithm& node) override;

      virtual void lower(const ast::Statement& node) override;

      virtual void lower(const ast::AssignmentStatement& statement) override;

      virtual void lower(const ast::BreakStatement& statement) override;

      virtual void lower(const ast::ForStatement& statement) override;

      virtual void lower(const ast::IfStatement& statement) override;

      virtual void lower(const ast::ReturnStatement& statement) override;

      virtual void lower(const ast::WhenStatement& statement) override;

      virtual void lower(const ast::WhileStatement& statement) override;

      virtual void lower(
          const ast::Equation& equation,
          bool initialEquation) override;

      virtual void lower(
          const ast::ForEquation& forEquation,
          bool initialEquation) override;

      /// }

    private:
      mlir::Operation* resolveSymbolName(
          llvm::StringRef name,
          mlir::Operation* currentScope,
          std::function<bool(mlir::Operation*)> filterFn);

    private:
      BridgeInterface* bridge;
  };
}

#endif // MARCO_CODEGEN_LOWERING_LOWERER_H
