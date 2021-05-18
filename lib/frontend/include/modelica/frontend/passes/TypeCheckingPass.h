#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <modelica/frontend/Pass.h>
#include <modelica/frontend/Symbol.hpp>

namespace modelica::frontend
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Call;
	class Class;
	class ClassContainer;
	class Constant;
	class DerFunction;
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

	class TypeChecker : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error run(Class& cls) final;

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

		[[nodiscard]] llvm::Error checkGenericOperation(Expression& expression);

		[[nodiscard]] llvm::Error checkAddOp(Expression& expression);
		[[nodiscard]] llvm::Error checkDifferentOp(Expression& expression);
		[[nodiscard]] llvm::Error checkDivOp(Expression& expression);
		[[nodiscard]] llvm::Error checkEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkGreaterOp(Expression& expression);
		[[nodiscard]] llvm::Error checkGreaterEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkIfElseOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLogicalAndOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLogicalOrOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLessOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLessEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkMemberLookupOp(Expression& expression);
		[[nodiscard]] llvm::Error checkMulOp(Expression& expression);
		[[nodiscard]] llvm::Error checkNegateOp(Expression& expression);
		[[nodiscard]] llvm::Error checkPowerOfOp(Expression& expression);
		[[nodiscard]] llvm::Error checkSubOp(Expression& expression);
		[[nodiscard]] llvm::Error checkSubscriptionOp(Expression& expression);

		private:
		llvm::ScopedHashTable<llvm::StringRef, Symbol> symbolTable;

		llvm::Expected<Type> typeFromSymbol(const Expression& exp);
	};

	template<>
	llvm::Error TypeChecker::run<Class>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<DerFunction>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Model>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Package>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Record>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Expression>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Array>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Call>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Constant>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Operation>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Tuple>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<AssignmentStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<BreakStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<ForStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<IfStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<ReturnStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<WhenStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createTypeCheckingPass();
}
