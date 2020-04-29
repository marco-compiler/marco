#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"
namespace modelica
{
	class TypeChecker
	{
		public:
		llvm::Error checkType(Class& cl, const SymbolTable& table);

		llvm::Error checkType(Expression& exp, const SymbolTable& table);
		llvm::Error checkType(Member& mem, const SymbolTable& table);
		llvm::Error checkType(Equation& eq, const SymbolTable& table);
		llvm::Error checkCall(Expression& call, const SymbolTable& table);
		llvm::Error checkOperation(Expression& call, const SymbolTable& table);
	};
}	 // namespace modelica
