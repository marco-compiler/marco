#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <marco/ast/Errors.h>
#include <marco/ast/Pass.h>
#include <marco/ast/Symbol.h>

namespace marco::ast
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Call;
	class Class;
	class Constant;
	class Equation;
	class Expression;
	class ForEquation;
	class ForStatement;
	class IfStatement;
	class Member;
	class Operation;
	class ReferenceAccess;
	class ReturnStatement;
	class StandardFunction;
	class Statement;
	class Tuple;
	class Type;
	class WhenStatement;
	class WhileStatement;

	class InlineExpanser : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using TranslationTable = llvm::ScopedHashTable<llvm::StringRef, Expression*>;

		InlineExpanser();

		llvm::Error run(llvm::ArrayRef<std::unique_ptr<Class>> classes) final;

		llvm::Error run(Algorithm& algorithm);

		template<typename T>
		[[nodiscard]] llvm::Error run(Class& cls);

		llvm::Error run(Equation& equation);

		template<typename T>
		[[nodiscard]] llvm::Error run(Expression& expression);

		llvm::Error run(ForEquation& forEquation);
		llvm::Error run(Induction& induction);
		llvm::Error run(Member& member);

		template<typename T>
		[[nodiscard]] llvm::Error run(Statement& statement);

		private:
		SymbolTable symbolTable;
		TranslationTable translationTable;
	};

	template<>
	llvm::Error InlineExpanser::run<Class>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<PartialDerFunction>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<Model>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<Package>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<Record>(Class& cls);

	template<>
	llvm::Error InlineExpanser::run<Expression>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<Array>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<Call>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<Constant>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<Operation>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<Tuple>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<RecordInstance>(Expression& expression);

	template<>
	llvm::Error InlineExpanser::run<AssignmentStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<BreakStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<ForStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<IfStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<ReturnStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<WhenStatement>(Statement& statement);

	template<>
	llvm::Error InlineExpanser::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createInliningPass();
}

