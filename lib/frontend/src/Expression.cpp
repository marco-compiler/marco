#include "modelica/frontend/Expression.hpp"

#include "llvm/ADT/StringRef.h"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
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
		args.emplace_back(std::make_unique<Expression>(arg));

	return Expression(
			Type::unkown(),
			Call(std::make_unique<Expression>(move(fun)), move(args)));
}

void Expression::dump(llvm::raw_ostream& OS, size_t nestNevel) const
{
	if (isA<Operation>())
	{
		get<Operation>().dump(OS, nestNevel);
		return;
	}

	if (isA<Constant>())
	{
		get<Constant>().dump(OS, nestNevel);
		return;
	}

	if (isA<ReferenceAccess>())
	{
		get<ReferenceAccess>().dump(OS, nestNevel);
		return;
	}

	if (isA<Call>())
	{
		get<Call>().dump(OS, nestNevel);
		return;
	}
	assert(false && "unrechable");
}

StringRef operationToString(OperationKind kind)
{
	switch (kind)
	{
		case OperationKind::negate:
			return "negate";
		case OperationKind::add:
			return "add";
		case OperationKind::subtract:
			return "subtract";
		case OperationKind::multiply:
			return "multiply";
		case OperationKind::divide:
			return "divide";
		case OperationKind::ifelse:
			return "ifelse";
		case OperationKind::greater:
			return "greater";
		case OperationKind::greaterEqual:
			return "greaterEqual";
		case OperationKind::equal:
			return "equal";
		case OperationKind::different:
			return "different";
		case OperationKind::lessEqual:
			return "lessEqual";
		case OperationKind::less:
			return "less";
		case OperationKind::land:
			return "land";
		case OperationKind::lor:
			return "lor";
		case OperationKind::subscription:
			return "subscription";
		case OperationKind::memberLookup:
			return "memberLookup";
		case OperationKind::powerOf:
			return "powerOf";
	}
	assert(false && "unreachable");
	return "";
}

void Expression::Operation::dump(llvm::raw_ostream& OS, size_t nestLevel) const
{
	OS.indent(nestLevel);
	OS << "Operation " << operationToString(kind) << " args:\n";
	for (const auto& arg : arguments)
	{
		arg.dump(OS, nestLevel + 1);
		OS << '\n';
	}
}
