#include "modelica/frontend/Expression.hpp"

#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

bool Expression::Operation::operator==(const Operation& other) const
{
	if (kind != other.kind)
		return false;
	if (arguments.size() != other.arguments.size())
		return false;

	return arguments == other.arguments;
}

[[nodiscard]] Expression modelica::makeCall(
		Expression fun, SmallVector<Expression, 3> exps)
{
	SmallVector<Call::UniqueExpr, 3> args;
	for (auto& arg : exps)
		args.emplace_back(make_unique<Expression>(arg));

	return Expression(
			Type::unkown(), Call(make_unique<Expression>(move(fun)), move(args)));
}
