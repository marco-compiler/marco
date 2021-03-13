#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Symbol.hpp>

namespace modelica
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Call;
	class Class;
	class ClassContainer;
	class Constant;
	class Equation;
	class Expression;
	class ForEquation;
	class ForStatement;
	class Function;
	class IfStatement;
	class Member;
	class Operation;
	class ReferenceAccess;
	class ReturnStatement;
	class Statement;
	class Tuple;
	class WhenStatement;
	class WhileStatement;

	class TypeChecker
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error check(ClassContainer& cls);
		llvm::Error check(Class& model);
		llvm::Error check(Function& function);
		llvm::Error check(Member& member);
		llvm::Error check(Algorithm& algorithm);
		llvm::Error check(Statement& statement);
		llvm::Error check(AssignmentStatement& statement);
		llvm::Error check(IfStatement& statement);
		llvm::Error check(ForStatement& statement);
		llvm::Error check(WhileStatement& statement);
		llvm::Error check(WhenStatement& statement);
		llvm::Error check(BreakStatement& statement);
		llvm::Error check(ReturnStatement& statement);
		llvm::Error check(Equation& equation);
		llvm::Error check(ForEquation& forEquation);

		template<typename T>
		llvm::Error check(Expression& expression);

		private:
		llvm::ScopedHashTable<llvm::StringRef, Symbol> symbolTable;

		llvm::Expected<Type> typeFromSymbol(const Expression& exp);
	};

	template<>
	llvm::Error TypeChecker::check<Expression>(Expression& expression);

	template<>
	llvm::Error TypeChecker::check<Operation>(Expression& expression);

	template<>
	llvm::Error TypeChecker::check<Constant>(Expression& expression);

	template<>
	llvm::Error TypeChecker::check<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error TypeChecker::check<Call>(Expression& expression);

	template<>
	llvm::Error TypeChecker::check<Tuple>(Expression& expression);
}	 // namespace modelica
