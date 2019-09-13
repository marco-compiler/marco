#include "modelica/AST/Expr.hpp"

#include "modelica/ParserErrors.hpp"

using namespace modelica;

[[nodiscard]] llvm::Error IfElseExpr::isConsistent() const
{
	if (size() < 1)
		return llvm::make_error<ChoiseNotFound>();

	if (getType() == Type(BuiltinType::Unknown))
		return llvm::Error::success();

	for (int i = 0; i < size(); i = i + 2)
	{
		if (at(i + 1)->getType() != getType())
			return llvm::make_error<BranchesTypeDoNotMatch>();
		if (at(i)->getType() != Type(BuiltinType::Boolean))
			return llvm::make_error<IncompatibleType>(getRange().getBegin());
	}
	if (getType() == Type(BuiltinType::None))
		return llvm::make_error<IncompatibleType>(getRange().getBegin());
	return llvm::Error::success();
}

[[nodiscard]] llvm::Error BinaryExpr::isConsistent() const
{
	if (getLeftHand()->getType() == Type(BuiltinType::None))
		return llvm::make_error<IncompatibleType>(getRange().getBegin());
	if (getRightHand()->getType() == Type(BuiltinType::None))
		return llvm::make_error<IncompatibleType>(getRange().getBegin());

	return llvm::Error::success();
}

[[nodiscard]] llvm::Error UnaryExpr::isConsistent() const
{
	if (getOperand()->getType() == Type(BuiltinType::None))
		return llvm::make_error<IncompatibleType>(getRange().getEnd());
	return llvm::Error::success();
}

[[nodiscard]] llvm::Error ExprList::isConsistent() const
{
	return llvm::Error::success();
}
