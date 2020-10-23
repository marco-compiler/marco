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
		llvm::Error checkType(Class& cl, const SymbolTable& table);
		llvm::Error checkType(Expression& exp, const SymbolTable& table);
		llvm::Error checkType(Member& mem, const SymbolTable& table);
		llvm::Error checkType(Equation& eq, const SymbolTable& table);
		llvm::Error checkType(ForEquation& eq, const SymbolTable& table);
		llvm::Error checkCall(Expression& call, const SymbolTable& table);
		llvm::Error checkOperation(Expression& call, const SymbolTable& table);
	};
}	 // namespace modelica
