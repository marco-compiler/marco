#pragma once

#include "llvm/Support/Error.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
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
		llvm::Error foldExpression(Expression& exp, const SymbolTable& table);
	};
}	 // namespace modelica
