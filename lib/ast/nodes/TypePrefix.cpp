#include "marco/ast/AST.h"

using namespace marco::ast;

namespace marco::ast
{
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const ParameterQualifier& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(ParameterQualifier qualifier)
	{
		switch (qualifier)
		{
			case ParameterQualifier::discrete:
				return "discrete";
			case ParameterQualifier::parameter:
				return "parameter";
			case ParameterQualifier::constant:
				return "constant";
			case ParameterQualifier::none:
				return "none";
		}

		return "unexpected";
	}

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const IOQualifier& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(IOQualifier qualifier)
	{
		switch (qualifier)
		{
			case IOQualifier::input:
				return "input";
			case IOQualifier::output:
				return "output";
			case IOQualifier::none:
				return "none";
		}

		return "unexpected";
	}
}

TypePrefix::TypePrefix(
		ParameterQualifier parameterQualifier, IOQualifier ioQualifier)
		: parameterQualifier(parameterQualifier), ioQualifier(ioQualifier)
{
}

void TypePrefix::print(llvm::raw_ostream& os, size_t indents) const
{
	os << "Prefix: ";

	if (parameterQualifier == ParameterQualifier::none &&
			ioQualifier == IOQualifier::none)
	{
		os << "none";
	}
	else
	{
		bool space = false;

		if (parameterQualifier != ParameterQualifier::none)
		{
			os << parameterQualifier;
			space = true;
		}

		if (ioQualifier != IOQualifier::none)
		{
			if (space)
				os << " ";

			os << ioQualifier;
		}
	}

	os.indent(indents);
}

bool TypePrefix::isParameter() const
{
	return parameterQualifier == ParameterQualifier::parameter ||
				 parameterQualifier == ParameterQualifier::constant;
}

bool TypePrefix::isInput() const { return ioQualifier == IOQualifier::input; }

bool TypePrefix::isOutput() const { return ioQualifier == IOQualifier::output; }

TypePrefix TypePrefix::none()
{
	return TypePrefix(ParameterQualifier::none, IOQualifier::none);
}