#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

using Container = Operation::Container;

raw_ostream& modelica::operator<<(raw_ostream& stream, const OperationKind& obj)
{
	return stream << toString(obj);
}

string modelica::toString(OperationKind operation)
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
	os << "operation kind: " << kind << "\n";

	os.indent(indents);
	os << "args:\n";

	for (const auto& arg : arguments)
		arg.dump(os, indents + 1);
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

Operation::iterator Operation::begin() { return arguments.begin(); }

Operation::const_iterator Operation::begin() const { return arguments.begin(); }

Operation::iterator Operation::end() { return arguments.end(); }

Operation::const_iterator Operation::end() const { return arguments.end(); }
