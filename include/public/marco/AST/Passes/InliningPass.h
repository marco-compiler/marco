#pragma once

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::ast
{

	class InlineExpanser : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using TranslationTable = llvm::ScopedHashTable<llvm::StringRef, Expression*>;

		InlineExpanser(diagnostic::DiagnosticEngine& diagnostics);

		bool run(std::unique_ptr<Class>& cls) override;

		bool run(Algorithm& algorithm);

		template<typename T>
		bool run(Class& cls);

		bool run(Equation& equation);

		template<typename T>
		bool run(Expression& expression);

		bool run(ForEquation& forEquation);
		bool run(Induction& induction);
		bool run(Member& member);

		template<typename T>
		bool run(Statement& statement);

		private:
		SymbolTable symbolTable;
		TranslationTable translationTable;
	};

	template<>
	bool InlineExpanser::run<Class>(Class& cls);

	template<>
	bool InlineExpanser::run<PartialDerFunction>(Class& cls);

	template<>
	bool InlineExpanser::run<StandardFunction>(Class& cls);

	template<>
	bool InlineExpanser::run<Model>(Class& cls);

	template<>
	bool InlineExpanser::run<Package>(Class& cls);

	template<>
	bool InlineExpanser::run<Record>(Class& cls);

	template<>
	bool InlineExpanser::run<Expression>(Expression& expression);

	template<>
	bool InlineExpanser::run<Array>(Expression& expression);

	template<>
	bool InlineExpanser::run<Call>(Expression& expression);

	template<>
	bool InlineExpanser::run<Constant>(Expression& expression);

	template<>
	bool InlineExpanser::run<Operation>(Expression& expression);

	template<>
	bool InlineExpanser::run<ReferenceAccess>(Expression& expression);

	template<>
	bool InlineExpanser::run<Tuple>(Expression& expression);

	template<>
	bool InlineExpanser::run<RecordInstance>(Expression& expression);

	template<>
	bool InlineExpanser::run<AssignmentStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<BreakStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<ForStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<IfStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<ReturnStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<WhenStatement>(Statement& statement);

	template<>
	bool InlineExpanser::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createInliningPass(diagnostic::DiagnosticEngine& diagnostics);
}

