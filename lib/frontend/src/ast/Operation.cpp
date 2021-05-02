#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>
#include <numeric>

using namespace modelica;
using namespace frontend;

using Container = Operation::Container;

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const OperationKind& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(OperationKind operation)
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
}

Operation::Operation(SourcePosition location, OperationKind kind, Container args)
		: location(std::move(location)),
			arguments(std::move(args)),
			kind(kind)
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

void Operation::dump() const { dump(llvm::outs(), 0); }

void Operation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "operation kind: " << kind << "\n";

	os.indent(indents);
	os << "args:\n";

	for (const auto& arg : arguments)
		arg.dump(os, indents + 1);
}

SourcePosition Operation::getLocation() const
{
	return location;
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

size_t Operation::size() const { return arguments.size(); }

Operation::iterator Operation::begin() { return arguments.begin(); }

Operation::const_iterator Operation::begin() const { return arguments.begin(); }

Operation::iterator Operation::end() { return arguments.end(); }

Operation::const_iterator Operation::end() const { return arguments.end(); }

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Operation& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Operation& obj)
	{
		switch (obj.getKind())
		{
			case OperationKind::negate:
				return "(not" + toString(obj[0]) + ")";

			case OperationKind::add:
				return "(" +
							 accumulate(obj.begin(), obj.end(), std::string(),
													[](const std::string& result, const Expression& element)
													{
														std::string str = toString(element);
														return result.empty() ? str : result + " + " + str;
													})
							 + ")";

			case OperationKind::subtract:
				return "(" +
							 accumulate(obj.begin(), obj.end(), std::string(),
													[](const std::string& result, const Expression& element)
													{
														std::string str = toString(element);
														return result.empty() ? str : result + " - " + str;
													})
							 + ")";

			case OperationKind::multiply:
				return "(" +
							 accumulate(obj.begin(), obj.end(), std::string(),
													[](const std::string& result, const Expression& element)
													{
														std::string str = toString(element);
														return result.empty() ? str : result + " * " + str;
													})
							 + ")";

			case OperationKind::divide:
				return "(" +
							 accumulate(obj.begin(), obj.end(), std::string(),
													[](const std::string& result, const Expression& element)
													{
														std::string str = toString(element);
														return result.empty() ? str : result + " / " + str;
													})
							 + ")";

			case OperationKind::ifelse:
				return "(" + toString(obj[0]) + " ? " + toString(obj[1]) + " : " + toString(obj[2]) + ")";

			case OperationKind::greater:
				return "(" + toString(obj[0]) + " > " + toString(obj[1]) + ")";

			case OperationKind::greaterEqual:
				return "(" + toString(obj[0]) + " >= " + toString(obj[1]) + ")";

			case OperationKind::equal:
				return "(" + toString(obj[0]) + " == " + toString(obj[1]) + ")";

			case OperationKind::different:
				return "(" + toString(obj[0]) + " != " + toString(obj[1]) + ")";

			case OperationKind::lessEqual:
				return "(" + toString(obj[0]) + " <= " + toString(obj[1]) + ")";

			case OperationKind::less:
				return "(" + toString(obj[0]) + " < " + toString(obj[1]) + ")";

			case OperationKind::land:
				return "(" + toString(obj[0]) + " && " + toString(obj[1]) + ")";

			case OperationKind::lor:
				return "(" + toString(obj[0]) + " || " + toString(obj[1]) + ")";

			case OperationKind::subscription:
				return "(" + toString(obj[0]) +
							 accumulate(++obj.begin(), obj.end(), std::string(),
													[](const std::string& result, const Expression& element)
													{
														std::string str = toString(element);
														return result + "[" + str + "]";
													}) +
							 ")";

			case OperationKind::memberLookup:
				return "unknown";

			case OperationKind::powerOf:
				return "(" + toString(obj[0]) + " ^ " + toString(obj[1]) + ")";
		}

		return "unknown";
	}
}
