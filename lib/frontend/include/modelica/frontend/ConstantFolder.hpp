#pragma once

#include <llvm/Support/Error.h>

#include "Class.hpp"
#include "Equation.hpp"
#include "Expression.hpp"
#include "Member.hpp"
#include "ReferenceAccess.hpp"
#include "SymbolTable.hpp"

namespace modelica
{
	class ConstantFolder
	{
		public:
		llvm::Error fold(Equation& eq, const SymbolTable& table);
		llvm::Error fold(ForEquation& eq, const SymbolTable& table);
		llvm::Error fold(Expression& exp, const SymbolTable& table);
		llvm::Error fold(Class& cl, const SymbolTable& table);
		llvm::Error fold(Member& mem, const SymbolTable& table);
		llvm::Error fold(Call& call, const SymbolTable& table);
		llvm::Error foldReference(Expression& exp, const SymbolTable& table);
		llvm::Expected<Expression> foldExpression(
				Expression& exp, const SymbolTable& table);
	};
}	 // namespace modelica
