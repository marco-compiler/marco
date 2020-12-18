#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Class.hpp>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Function.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/Symbol.hpp>
#include <modelica/frontend/Type.hpp>

namespace modelica
{
	class TypeChecker
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error check(ClassContainer& cls);
		llvm::Error check(Function& function);
		llvm::Error check(Class& model);
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

		/**
		 * Get the symbol table in use.
		 * Should be used only for unit testing.
		 *
		 * @return symbol table
		 */
		SymbolTable& getSymbolTable()
		{
			return symbolTable;
		}

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
