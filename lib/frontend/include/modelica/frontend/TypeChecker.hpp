#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Class.hpp>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Function.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/frontend/Type.hpp>

namespace modelica
{
	class TypeChecker
	{
		public:
		llvm::Error checkType(ClassContainer& cls, const SymbolTable& table);
		llvm::Error checkType(Function& function, const SymbolTable& table);
		llvm::Error checkType(Class& model, const SymbolTable& table);

		template<typename T>
		llvm::Error checkType(Expression& exp, const SymbolTable& table);

		llvm::Error checkType(Algorithm& algorithm, const SymbolTable& table);
		llvm::Error checkType(Statement& statement, const SymbolTable& table);
		llvm::Error checkType(
				AssignmentStatement& statement, const SymbolTable& table);
		llvm::Error checkType(IfStatement& statement, const SymbolTable& table);
		llvm::Error checkType(
				IfStatement::Block& statement, const SymbolTable& table);
		llvm::Error checkType(ForStatement& statement, const SymbolTable& table);
		llvm::Error checkType(BreakStatement& statement, const SymbolTable& table);
		llvm::Error checkType(ReturnStatement& statement, const SymbolTable& table);

		llvm::Error checkType(Member& mem, const SymbolTable& table);
		llvm::Error checkType(Equation& eq, const SymbolTable& table);
		llvm::Error checkType(ForEquation& eq, const SymbolTable& table);
	};

	template<>
	llvm::Error TypeChecker::checkType<Expression>(
			Expression& expression, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<Operation>(
			Expression& expression, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<Constant>(
			Expression& expression, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<ReferenceAccess>(
			Expression& expression, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<Call>(
			Expression& expression, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<Tuple>(
			Expression& expression, const SymbolTable& table);
}	 // namespace modelica
