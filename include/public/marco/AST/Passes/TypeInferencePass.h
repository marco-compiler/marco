#ifndef MARCO_AST_PASSES_TYPEINFERENCEPASS_H
#define MARCO_AST_PASSES_TYPEINFERENCEPASS_H

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "marco/AST/BuiltInFunction.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::ast
{
  class TypeInferencePass : public Pass
  {
    public:
      using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
      using SymbolTableScope = SymbolTable::ScopeTy;
    
      TypeInferencePass(diagnostic::DiagnosticEngine& diagnostics);

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
      llvm::StringMap<std::unique_ptr<BuiltInFunction>> builtInFunctions;
  };
  
  template<>
  bool TypeInferencePass::run<Class>(Class& cls);
  
  template<>
  bool TypeInferencePass::run<PartialDerFunction>(Class& cls);

  template<>
  bool TypeInferencePass::run<StandardFunction>(Class& cls);

  template<>
  bool TypeInferencePass::run<Model>(Class& cls);

  template<>
  bool TypeInferencePass::run<Package>(Class& cls);

  template<>
  bool TypeInferencePass::run<Record>(Class& cls);

  template<>
  bool TypeInferencePass::run<Expression>(Expression& expression);

  template<>
  bool TypeInferencePass::run<Array>(Expression& expression);

  template<>
  bool TypeInferencePass::run<Call>(Expression& expression);

  template<>
  bool TypeInferencePass::run<Constant>(Expression& expression);

  template<>
  bool TypeInferencePass::run<Operation>(Expression& expression);

  template<>
  bool TypeInferencePass::run<ReferenceAccess>(Expression& expression);

  template<>
  bool TypeInferencePass::run<Tuple>(Expression& expression);

  template<>
  bool TypeInferencePass::run<RecordInstance>(Expression& expression);
  
  template<>
  bool TypeInferencePass::run<AssignmentStatement>(Statement& statement);
  
  template<>
  bool TypeInferencePass::run<BreakStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<ForStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<IfStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<ReturnStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<WhenStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<WhileStatement>(Statement& statement);

  template<>
  bool TypeInferencePass::run<Argument>(Argument& argument);

  template<>
  bool TypeInferencePass::run<ElementModification>(Argument& argument);

  template<>
  bool TypeInferencePass::run<ElementRedeclaration>(Argument& argument);

  template<>
  bool TypeInferencePass::run<ElementReplaceable>(Argument& argument);
  
  std::unique_ptr<Pass> createTypeInferencePass(diagnostic::DiagnosticEngine& diagnostics);
}

#endif // MARCO_AST_PASSES_TYPEINFERENCEPASS_H
