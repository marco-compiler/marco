#ifndef MARCO_CODEGEN_LOWERING_LOWERER_H
#define MARCO_CODEGEN_LOWERING_LOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering
{
  class Lowerer : public BridgeInterface
  {
    public:
      using VariablesScope = LoweringContext::VariablesScope;
      using LookupScopeGuard = LoweringContext::LookupScopeGuard;

      explicit Lowerer(BridgeInterface* bridge);

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

      std::set<llvm::StringRef>& getDeclaredVariables();

      // Helper to initialize declaredSymbols, adding all the declared symbols visible from scope 
      // of the requested type.
      template<typename... T>
      void initializeDeclaredSymbols(
          mlir::Operation* scope, 
          std::set<std::string> &declaredSymbols) 
      {
        return initializeDeclaredSymbols(scope, declaredSymbols, [](mlir::Operation* op) {
          return mlir::isa<T...>(op);
        });
      }

      void initializeDeclaredSymbols(mlir::Operation* scope, std::set<std::string> &declaredSymbols);

      // Helper to initialize declaredVars, adding all the declared variables visible from the current scope.
      void initializeDeclaredVars(std::set<std::string> &declaredVars);

      mlir::Operation* getLookupScope();

      void pushLookupScope(mlir::Operation* lookupScope);

      mlir::Operation* getClass(const ast::Class& cls);

      mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation* symbol);

      mlir::Operation* resolveClassName(
          llvm::StringRef name,
          mlir::Operation* currentScope);

      std::optional<mlir::Operation*> resolveType(
          const ast::UserDefinedType& type,
          mlir::Operation* lookupScope);

      mlir::Operation* resolveTypeFromRoot(mlir::SymbolRefAttr name);

      mlir::Operation* resolveSymbolName(
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

      std::optional<Reference> lookupVariable(llvm::StringRef name);

      void insertVariable(llvm::StringRef name, Reference reference);

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

      [[nodiscard]] virtual bool declareVariables(const ast::Class& node) override;

      [[nodiscard]] virtual bool declareVariables(const ast::Model& model) override;

      [[nodiscard]] virtual bool declareVariables(const ast::Package& package) override;

      virtual void declareVariables(
          const ast::PartialDerFunction& function) override;

      [[nodiscard]] virtual bool declareVariables(const ast::Record& record) override;

      [[nodiscard]] virtual bool declareVariables(
          const ast::StandardFunction& function) override;

      [[nodiscard]] virtual bool declare(const ast::Member& node) override;

      [[nodiscard]] virtual bool lower(const ast::Class& node) override;

      [[nodiscard]] virtual bool lower(const ast::Model& node) override;

      [[nodiscard]] virtual bool lower(const ast::Package& node) override;

      virtual void lower(const ast::PartialDerFunction& node) override;

      [[nodiscard]] virtual bool lower(const ast::Record& node) override;

      [[nodiscard]] virtual bool 
          lower(const ast::StandardFunction& node) override;

      [[nodiscard]] virtual bool 
          lowerClassBody(const ast::Class& node) override;

      [[nodiscard]] virtual bool createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) override;

      [[nodiscard]] virtual bool lowerStartAttribute(
          mlir::SymbolRefAttr variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) override;

      virtual std::optional<Results> lower(const ast::Expression& expression) override;

      virtual std::optional<Results> lower(const ast::ArrayGenerator& node) override;

      virtual std::optional<Results> lower(const ast::Call& node) override;

      virtual Results lower(const ast::Constant& constant) override;

      virtual std::optional<Results> lower(const ast::Operation& operation) override;

      virtual std::optional<Results> lower(
          const ast::ComponentReference& componentReference) override;

      virtual std::optional<Results> lower(const ast::Tuple& tuple) override;

      virtual std::optional<Results> lower(const ast::Subscript& subscript) override;

      [[nodiscard]] virtual bool lower(const ast::EquationSection& node) override;

      [[nodiscard]] virtual bool lower(const ast::Equation& equation) override;

      [[nodiscard]] virtual bool lower(const ast::EqualityEquation& equation) override;

      [[nodiscard]] virtual bool lower(const ast::ForEquation& equation) override;

      virtual void lower(const ast::IfEquation& equation) override;

      virtual void lower(const ast::WhenEquation& equation) override;

      [[nodiscard]] virtual bool lower(const ast::Algorithm& node) override;

      [[nodiscard]] virtual bool lower(const ast::Statement& node) override;

      [[nodiscard]] virtual bool lower(const ast::AssignmentStatement& statement) override;

      virtual void lower(const ast::BreakStatement& statement) override;

      [[nodiscard]] virtual bool lower(const ast::ForStatement& statement) override;

      [[nodiscard]] virtual bool lower(const ast::IfStatement& statement) override;

      virtual void lower(const ast::ReturnStatement& statement) override;

      virtual void lower(const ast::WhenStatement& statement) override;

      [[nodiscard]] virtual bool 
          lower(const ast::WhileStatement& statement) override;

      virtual void emitIdentifierError(IdentifierError::IdentifierType identifierType, std::string name, 
                                       const std::set<std::string> &declaredIdentifiers, 
                                       unsigned int line, unsigned int column) override;
      virtual void emitError(const std::string &error) override;

      /// }

    private:
      mlir::Operation* resolveSymbolName(
          llvm::StringRef name,
          mlir::Operation* currentScope,
          llvm::function_ref<bool(mlir::Operation*)> filterFn);
      
      void initializeDeclaredSymbols(
          mlir::Operation* scope, 
          std::set<std::string> &declaredSymbols,
          llvm::function_ref<bool(mlir::Operation*)> filterFn);

    private:
      BridgeInterface* bridge;
  };
}

#endif // MARCO_CODEGEN_LOWERING_LOWERER_H
