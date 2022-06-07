#ifndef MARCO_AST_PASSES_STRUCTURALCONSTANTPROPAGATIONPASS_H
#define MARCO_AST_PASSES_STRUCTURALCONSTANTPROPAGATIONPASS_H

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::ast
{
  class StructuralConstantPropagationPass : public Pass
  {
    public:
      using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
      using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

      StructuralConstantPropagationPass(diagnostic::DiagnosticEngine& diagnostics);

      bool run(std::unique_ptr<Class>& cls) override;

      template<typename T>
      bool run(Class& cls);

      bool run(ForEquation& forEquation);

      bool run(Member& member);

      template<typename T>
      bool run(Expression& expression);

    private:
      SymbolTable symbolTable;
  };

  template<>
  bool StructuralConstantPropagationPass::run<Class>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<PartialDerFunction>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<StandardFunction>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<Model>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<Package>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<Record>(Class& cls);

  template<>
  bool StructuralConstantPropagationPass::run<Expression>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<Array>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<Call>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<Constant>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<Operation>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<ReferenceAccess>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<Tuple>(Expression& expression);

  template<>
  bool StructuralConstantPropagationPass::run<RecordInstance>(Expression& expression);

  std::unique_ptr<Pass> createStructuralConstantPropagationPass(diagnostic::DiagnosticEngine& diagnostics);
}

#endif // MARCO_AST_PASSES_STRUCTURALCONSTANTPROPAGATIONPASS_H
