#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/utils/IRange.hpp>

using namespace modelica;
using namespace std;
using namespace llvm;

using Container = Operation::Container;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const OperationKind& obj)
	{
		return stream << toString(obj);
	}

	string toString(OperationKind operation)
	{
		switch (operation)
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

		return "unexpected";
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

bool Operation::isLValue() const
{
	switch (kind)
	{
		case OperationKind::subscription:
			return arguments[0].isLValue();

		case OperationKind::memberLookup:
			return true;

		default:
			return false;
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

Expression::Expression(Tuple tuple): content(move(tuple)), type(Type::tuple())
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

	assert(false && "Unreachable");
}

bool Expression::isLValue() const
{
	if (isA<Operation>())
		return get<Operation>().isLValue();

	if (isA<ReferenceAccess>())
		return true;

	return false;
}

Type& Expression::getType() { return type; }

const Type& Expression::getType() const { return type; }

void Expression::setType(Type tp) { type = move(tp); }

Expression modelica::makeCall(Expression fun, llvm::ArrayRef<Expression> args)
{
	return Expression(Type::unknown(), Call(move(fun), move(args)));
}
