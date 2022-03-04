#ifndef MARCO_CODEGEN_LOWERINGBRIDGE_H
#define MARCO_CODEGEN_LOWERINGBRIDGE_H

#include "llvm/ADT/ScopedHashTable.h"
#include "marco/AST/AST.h"
#include "marco/AST/SymbolTable.h"
#include "marco/Codegen/Options.h"
#include "marco/Codegen/Reference.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/ModelicaBuilder.h"
#include "marco/Utils/SourcePosition.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace marco::codegen
{
  class NewLoweringBridge
  {
    private:
      template<typename T> using Container = llvm::SmallVector<T, 3>;

    public:
      NewLoweringBridge(mlir::MLIRContext& context, CodegenOptions options = CodegenOptions::getDefaultOptions());

      llvm::Optional<mlir::ModuleOp> run(llvm::ArrayRef<std::unique_ptr<ast::Class>> classes);

    private:
      mlir::Operation* lower(const ast::Class& cls);
      mlir::Operation* lower(const ast::PartialDerFunction& function);
      mlir::Operation* lower(const ast::StandardFunction& function);
      mlir::Operation* lower(const ast::Model& model);
      mlir::Operation* lower(const ast::Package& package);
      mlir::Operation* lower(const ast::Record& record);

      mlir::Type lower(const ast::Type& type, mlir::modelica::ArrayAllocationScope scope);
      mlir::Type lower(const ast::BuiltInType& type, mlir::modelica::ArrayAllocationScope scope);
      mlir::Type lower(const ast::PackedType& type, mlir::modelica::ArrayAllocationScope scope);
      mlir::Type lower(const ast::UserDefinedType& type, mlir::modelica::ArrayAllocationScope scope);

      template<typename Context>
      void lower(const ast::Member& member);

      void lower(const ast::Equation& equation);
      void lower(const ast::ForEquation& forEquation);

      void lower(const ast::Algorithm& algorithm);
      void lower(const ast::Statement& statement);
      void lower(const ast::AssignmentStatement& statement);
      void lower(const ast::IfStatement& statement);
      void lower(const ast::ForStatement& statement);
      void lower(const ast::WhileStatement& statement);
      void lower(const ast::WhenStatement& statement);
      void lower(const ast::BreakStatement& statement);
      void lower(const ast::ReturnStatement& statement);

      template<typename T>
      Container<Reference> lower(const ast::Expression& expression);

      /// The builder is a helper class to create IR inside a function. The
      /// builder is stateful, in particular it keeps an "insertion point":
      /// this is where the next operations will be introduced.
      mlir::modelica::ModelicaBuilder builder;

      /// The symbol table maps a variable name to a value in the current scope.
      /// Entering a function creates a new scope, and the function arguments
      /// are added to the mapping. When the processing of a function is
      /// terminated, the scope is destroyed and the mappings created in this
      /// scope are dropped.
      llvm::ScopedHashTable<llvm::StringRef, Reference> symbolTable;

      // The stack represent the list of the nested scope names in which the
      // lowerer currently is.
      //std::deque<llvm::StringRef> scopes;

      /// Apply a binary operation to a list of values.
      ///
      /// @param args      arguments
      /// @param callback  callback that should process the current args and return a result
      /// @return folded value
      mlir::Value foldBinaryOperation(
          llvm::ArrayRef<mlir::Value> args,
          std::function<mlir::Value(mlir::Value, mlir::Value)> callback);

      /// Lower the arguments of an operation.
      ///
      /// @param operation operation whose arguments have to be lowered
      /// @return lowered args
      Container<mlir::Value> lowerOperationArgs(const ast::Operation& operation);

      /// Helper to convert an AST location to a MLIR location.
      ///
      /// @param location frontend location
      /// @return MLIR location
      mlir::Location loc(SourcePosition location);

      /// Helper to convert an AST location to a MLIR location.
      ///
      /// @param location frontend location
      /// @return MLIR location
      mlir::Location loc(SourceRange location);

    private:
      CodegenOptions options;
  };

  template<>
  void NewLoweringBridge::lower<ast::Model>(
      const ast::Member& member);

  template<>
  void NewLoweringBridge::lower<ast::Function>(
      const ast::Member& member);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Expression>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Operation>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Constant>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::ReferenceAccess>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Call>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Tuple>(
      const ast::Expression& expression);

  template<>
  NewLoweringBridge::Container<Reference> NewLoweringBridge::lower<ast::Array>(
      const ast::Expression& expression);
}

#endif // MARCO_CODEGEN_LOWERINGBRIDGE_H
