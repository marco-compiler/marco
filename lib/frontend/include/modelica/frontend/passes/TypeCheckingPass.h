#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <modelica/frontend/Pass.h>
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
	class Type;
	class WhenStatement;
	class WhileStatement;

	class TypeChecker: public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error run(ClassContainer& cls);
		llvm::Error run(Class& model);
		llvm::Error run(Function& function);
		llvm::Error run(Package& package);
		llvm::Error run(Record& record);
		llvm::Error run(Member& member);
		llvm::Error run(Algorithm& algorithm);
		llvm::Error run(Statement& statement);
		llvm::Error run(AssignmentStatement& statement);
		llvm::Error run(IfStatement& statement);
		llvm::Error run(ForStatement& statement);
		llvm::Error run(WhileStatement& statement);
		llvm::Error run(WhenStatement& statement);
		llvm::Error run(BreakStatement& statement);
		llvm::Error run(ReturnStatement& statement);
		llvm::Error run(Equation& equation);
		llvm::Error run(ForEquation& forEquation);

		template<typename T>
		llvm::Error run(Expression& expression);

		private:
		llvm::ScopedHashTable<llvm::StringRef, Symbol> symbolTable;

		llvm::Expected<Type> typeFromSymbol(const Expression& exp);
	};

	template<>
	llvm::Error TypeChecker::run<Expression>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Operation>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Constant>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Call>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Tuple>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Array>(Expression& expression);

	std::unique_ptr<Pass> createTypeCheckingPass();
}	 // namespace modelica
