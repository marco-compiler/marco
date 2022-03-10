#ifndef MARCO_CODEGEN_LOWERINGBRIDGE_H
#define MARCO_CODEGEN_LOWERINGBRIDGE_H

#include "llvm/ADT/ScopedHashTable.h"
#include "marco/AST/AST.h"
#include "marco/AST/SymbolTable.h"
#include "marco/Codegen/Lowering/Reference.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/Options.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/ModelicaBuilder.h"
#include "marco/Utils/SourcePosition.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace marco::codegen::lowering
{
  class NewLoweringBridge
  {
    private:
      friend class FunctionCallBridge;
      friend class OperationBridge;

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
      Results lower(const ast::Expression& expression);

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

      /// Helper to convert an AST location to a MLIR location.
      mlir::Location loc(SourcePosition location);

      /// Helper to convert an AST location range to a MLIR location.
      mlir::Location loc(SourceRange location);

    private:
      CodegenOptions options;
  };

  template<>
  void NewLoweringBridge::lower<ast::Model>(const ast::Member& member);

  template<>
  void NewLoweringBridge::lower<ast::Function>(const ast::Member& member);

  template<>
  Results NewLoweringBridge::lower<ast::Expression>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::Operation>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::Constant>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::ReferenceAccess>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::Call>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::Tuple>(const ast::Expression& expression);

  template<>
  Results NewLoweringBridge::lower<ast::Array>(const ast::Expression& expression);
}

#endif // MARCO_CODEGEN_LOWERINGBRIDGE_H