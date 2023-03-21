#ifndef MARCO_AST_PASSES_SEMANTICANALYSISPASS_H
#define MARCO_AST_PASSES_SEMANTICANALYSISPASS_H

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "marco/AST/BuiltInFunctions.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::ast
{
  class SemanticAnalysisPass : public Pass
  {
    public:
      using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
      using SymbolTableScope = SymbolTable::ScopeTy;

      SemanticAnalysisPass(diagnostic::DiagnosticEngine& diagnostics);

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
      bool processComparisonOp(Expression& expression);

    private:
      SymbolTable symbolTable;
  };

  template<>
  bool SemanticAnalysisPass::run<Class>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<PartialDerFunction>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<StandardFunction>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<Model>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<Package>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<Record>(Class& cls);

  template<>
  bool SemanticAnalysisPass::run<Expression>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<Array>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<Call>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<Constant>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<Operation>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<ReferenceAccess>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<Tuple>(Expression& expression);

  template<>
  bool SemanticAnalysisPass::run<AssignmentStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<BreakStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<ForStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<IfStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<ReturnStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<WhenStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<WhileStatement>(Statement& statement);

  template<>
  bool SemanticAnalysisPass::run<Argument>(Argument& argument);

  template<>
  bool SemanticAnalysisPass::run<ElementModification>(Argument& argument);

  template<>
  bool SemanticAnalysisPass::run<ElementRedeclaration>(Argument& argument);

  template<>
  bool SemanticAnalysisPass::run<ElementReplaceable>(Argument& argument);

  std::unique_ptr<Pass> createSemanticAnalysisPass(diagnostic::DiagnosticEngine& diagnostics);
}

#endif // MARCO_AST_PASSES_SEMANTICANALYSISPASS_H
