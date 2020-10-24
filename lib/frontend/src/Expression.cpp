#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/utils/IRange.hpp>

using namespace modelica;
using namespace std;
using namespace llvm;

using Operation = Expression::Operation;
using Container = Operation::Container;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const OperationKind& obj)
	{
		if (obj == OperationKind::negate)
			stream << "negate";
		else if (obj == OperationKind::add)
			stream << "add";
		else if (obj == OperationKind::subtract)
			stream << "subtract";
		else if (obj == OperationKind::multiply)
			stream << "multiply";
		else if (obj == OperationKind::divide)
			stream << "divide";
		else if (obj == OperationKind::ifelse)
			stream << "ifelse";
		else if (obj == OperationKind::greater)
			stream << "greater";
		else if (obj == OperationKind::greaterEqual)
			stream << "greaterEqual";
		else if (obj == OperationKind::equal)
			stream << "equal";
		else if (obj == OperationKind::different)
			stream << "different";
		else if (obj == OperationKind::lessEqual)
			stream << "lessEqual";
		else if (obj == OperationKind::less)
			stream << "less";
		else if (obj == OperationKind::land)
			stream << "land";
		else if (obj == OperationKind::lor)
			stream << "lor";
		else if (obj == OperationKind::subscription)
			stream << "subscription";
		else if (obj == OperationKind::memberLookup)
			stream << "memberLookup";
		else if (obj == OperationKind::powerOf)
			stream << "powerOf";

		return stream;
	}
}	 // namespace modelica

Operation::Operation(OperationKind kind, Container args)
		: arguments(std::move(args)), kind(kind)
{
}

bool Operation::operator==(const Operation& other) const
{
	if (kind != other.kind)
		return false;

	if (arguments.size() != other.arguments.size())
		return false;

	return arguments == other.arguments;
}

bool Operation::operator!=(const Operation& other) const
{
	return !(*this == other);
}

Expression& Operation::operator[](size_t index) { return arguments[index]; }

const Expression& Operation::operator[](size_t index) const
{
	return arguments[index];
}

void Operation::dump() const { dump(outs(), 0); }

void Operation::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "Operation " << kind << " args:\n";

	for (const auto& arg : arguments)
	{
		arg.dump(os, indents + 1);
		os << '\n';
	}
}

OperationKind Operation::getKind() const { return kind; }

void Operation::setKind(OperationKind k) { kind = k; }

Container& Operation::getArguments() { return arguments; }

const Container& Operation::getArguments() const { return arguments; }

size_t Operation::argumentsCount() const { return arguments.size(); }

Container::iterator Operation::begin() { return arguments.begin(); }

Container::const_iterator Operation::begin() const { return arguments.begin(); }

Container::iterator Operation::end() { return arguments.end(); }

Container::const_iterator Operation::end() const { return arguments.end(); }

Expression::Expression(Type type, Constant constant)
		: content(move(constant)), type(move(type))
{
}

Expression::Expression(Type type, ReferenceAccess access)
		: content(move(access)), type(move(type))
{
}

Expression::Expression(Type type, Call call)
		: content(move(call)), type(move(type))
{
}

Expression::Expression(Type type, OperationKind kind, Operation::Container args)
		: content(Operation(kind, move(args))), type(move(type))
{
}

bool Expression::operator==(const Expression& other) const
{
	return type == other.type && content == other.content;
}

bool Expression::operator!=(const Expression& other) const
{
	return !(*this == other);
}

void Expression::dump() const { dump(outs(), 0); }

void Expression::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "type: ";
	getType().dump(os);

	if (isA<Operation>())
	{
		get<Operation>().dump(os, indents);
		return;
	}

	if (isA<Constant>())
	{
		get<Constant>().dump(os, indents);
		return;
	}

	if (isA<ReferenceAccess>())
	{
		get<ReferenceAccess>().dump(os, indents);
		return;
	}

	if (isA<Call>())
	{
		get<Call>().dump(os, indents);
		return;
	}
	assert(false && "unreachable");
}

bool Expression::isOperation() const { return isA<Operation>(); }

Operation& Expression::getOperation() { return get<Operation>(); }

const Operation& Expression::getOperation() const { return get<Operation>(); }

OperationKind Expression::getOperationKind() const
{
	return get<Operation>().getKind();
}

Constant& Expression::getConstant() { return get<Constant>(); }

const Constant& Expression::getConstant() const { return get<Constant>(); }

Type& Expression::getType() { return type; }

const Type& Expression::getType() const { return type; }

void Expression::setType(Type tp) { type = move(tp); }

Expression modelica::makeCall(Expression fun, llvm::ArrayRef<Expression> args)
{
	return Expression(Type::unknown(), Call(move(fun), move(args)));
}
