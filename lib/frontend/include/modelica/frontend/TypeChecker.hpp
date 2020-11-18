#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include "Class.hpp"
#include "Equation.hpp"
#include "Expression.hpp"
#include "Member.hpp"
#include "SymbolTable.hpp"
#include "Type.hpp"

namespace modelica
{
	class TypeChecker
	{
		public:
		template<ClassType T>
		llvm::Error checkType(Class& cl, const SymbolTable& table);

		template<typename T>
		llvm::Error checkType(Expression& exp, const SymbolTable& table);

		llvm::Error checkType(Algorithm& algorithm, const SymbolTable& table);
		llvm::Error checkType(Statement& statement, const SymbolTable& table);
		llvm::Error checkType(
				AssignmentStatement& statement, const SymbolTable& table);
		llvm::Error checkType(ForStatement& statement, const SymbolTable& table);
		llvm::Error checkType(IfStatement& statement, const SymbolTable& table);
		llvm::Error checkType(IfBlock& statement, const SymbolTable& table);

		llvm::Error checkType(Member& mem, const SymbolTable& table);
		llvm::Error checkType(Equation& eq, const SymbolTable& table);
		llvm::Error checkType(ForEquation& eq, const SymbolTable& table);
	};

	template<>
	llvm::Error TypeChecker::checkType<ClassType::Class>(
			Class& cl, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<ClassType::Function>(
			Class& cl, const SymbolTable& table);

	template<>
	llvm::Error TypeChecker::checkType<ClassType::Model>(
			Class& cl, const SymbolTable& table);

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
