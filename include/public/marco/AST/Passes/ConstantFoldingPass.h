#ifndef MARCO_AST_PASSES_CONSTANTFOLDINGPASS_H
#define MARCO_AST_PASSES_CONSTANTFOLDINGPASS_H

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "marco/AST/BuiltInFunction.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace marco::ast
{
	class ConstantFoldingPass : public Pass
	{
		public:
      ConstantFoldingPass(diagnostic::DiagnosticEngine& diagnostics);

      bool run(std::unique_ptr<Class>& cls) final;

      template<typename T>
      bool run(Class& cls);

      template<typename T>
      bool run(Expression& expression);

      template<OperationKind op>
      bool processOp(Expression& expression);

      template<typename T>
      bool run(Statement& statement);

      bool run(Equation& equation);

      bool run(ForEquation& forEquation);

      bool run(Induction& induction);

      bool run(Member& member);

      bool run(Algorithm& algorithm);
	};

  template<>
  bool ConstantFoldingPass::run<Class>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<PartialDerFunction>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<StandardFunction>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<Model>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<Package>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<Record>(Class& cls);

  template<>
  bool ConstantFoldingPass::run<Expression>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<Array>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<Call>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<Constant>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<Operation>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<ReferenceAccess>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<Tuple>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<RecordInstance>(Expression& expression);

  template<>
  bool ConstantFoldingPass::run<AssignmentStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<BreakStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<ForStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<IfStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<ReturnStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<WhenStatement>(Statement& statement);

  template<>
  bool ConstantFoldingPass::run<WhileStatement>(Statement& statement);

  std::unique_ptr<Pass> createConstantFoldingPass(diagnostic::DiagnosticEngine& diagnostics);
}

#endif // MARCO_AST_PASSES_CONSTANTFOLDINGPASS_H
