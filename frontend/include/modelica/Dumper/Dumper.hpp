#pragma once
#include "modelica/AST/Visitor.hpp"

namespace modelica
{
	std::unique_ptr<Statement> dump(
			std::unique_ptr<Statement> stmt, llvm::raw_ostream& OS = llvm::outs());
	std::unique_ptr<Declaration> dump(
			std::unique_ptr<Declaration> decl, llvm::raw_ostream& OS = llvm::outs());
	std::unique_ptr<Expr> dump(
			std::unique_ptr<Expr> exp, llvm::raw_ostream& OS = llvm::outs());
	std::unique_ptr<Equation> dump(
			std::unique_ptr<Equation> eq, llvm::raw_ostream& OS = llvm::outs());
}	// namespace modelica
