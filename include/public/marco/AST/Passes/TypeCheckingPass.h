#ifndef MARCO_AST_PASSES_TYPECHECKINGPASS_H
#define MARCO_AST_PASSES_TYPECHECKINGPASS_H

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <memory>

namespace marco::ast
{
  class TypeCheckingPass : public Pass
  {
    public:
      using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
      using SymbolTableScope = SymbolTable::ScopeTy;

      TypeCheckingPass(diagnostic::DiagnosticEngine& diagnostics);

      /// Get the symbol table. To be used just for testing purposes.
      SymbolTable& getSymbolTable();

      bool run(std::unique_ptr<Class>& cls) override;

      template<typename T>
      bool run(Class& cls);

      template<typename T>
      bool run(Expression& expression);

      template<OperationKind op>
      bool processOp(Expression& expression);

      template<typename T>
      bool run(Statement& statement);

      template<typename T>
      bool run(Argument& argument);

      bool run(Equation& equation);

      bool run(ForEquation& forEquation);

      bool run(Induction& induction);

      bool run(Member& member);

      bool run(Algorithm& algorithm);

      bool run(Modification& modification);

      bool run(ClassModification& classModification);

    private:
      SymbolTable symbolTable;
  };

  template<>
  bool TypeCheckingPass::run<Class>(Class& cls);

  template<>
  bool TypeCheckingPass::run<PartialDerFunction>(Class& cls);

  template<>
  bool TypeCheckingPass::run<StandardFunction>(Class& cls);

  template<>
  bool TypeCheckingPass::run<Model>(Class& cls);

  template<>
  bool TypeCheckingPass::run<Package>(Class& cls);

  template<>
  bool TypeCheckingPass::run<Record>(Class& cls);

  template<>
  bool TypeCheckingPass::run<Expression>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<Array>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<Call>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<Constant>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<Operation>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<ReferenceAccess>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<Tuple>(Expression& expression);

  template<>
  bool TypeCheckingPass::run<AssignmentStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<BreakStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<ForStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<IfStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<ReturnStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<WhenStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<WhileStatement>(Statement& statement);

  template<>
  bool TypeCheckingPass::run<Argument>(Argument& argument);

  template<>
  bool TypeCheckingPass::run<ElementModification>(Argument& argument);

  template<>
  bool TypeCheckingPass::run<ElementRedeclaration>(Argument& argument);

  template<>
  bool TypeCheckingPass::run<ElementReplaceable>(Argument& argument);

  std::unique_ptr<Pass> createTypeCheckingPass(diagnostic::DiagnosticEngine& diagnostics);
}

#endif // MARCO_AST_PASSES_TYPECHECKINGPASS_H
