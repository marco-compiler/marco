#include "modelica/AST/Expr.hpp"

#include <numeric>

#include "modelica/ParserErrors.hpp"

using namespace modelica;
using namespace std;

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

ExprList::ExprList(
		SourceRange location,
		Type type,
		ExprKind kind,
		std::vector<UniqueExpr> exprs)
		: Expr(std::move(location), type, kind), expressions(std::move(exprs))
{
	assert(std::accumulate(	// NOLINT
			std::begin(expressions),
			std::end(expressions),
			true,
			[](bool val, const auto& next) { return val && next != nullptr; }));
}

ExprList::ExprList(SourceRange location, std::vector<UniqueExpr> exprs)
		: Expr(
					std::move(location),
					Type(BuiltinType::None),
					ExprKind::ExpressionList),
			expressions(std::move(exprs))
{
	assert(std::accumulate(	// NOLINT
			std::begin(expressions),
			std::end(expressions),
			true,
			[](bool val, const auto& next) { return val && next != nullptr; }));
}
