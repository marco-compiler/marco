#include <modelica/frontend/AST.h>

using namespace modelica;

llvm::raw_ostream& modelica::operator<<(
		llvm::raw_ostream& stream, const ParameterQualifier& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(ParameterQualifier qualifier)
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

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const IOQualifier& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(IOQualifier qualifier)
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

TypePrefix::TypePrefix(
		ParameterQualifier parameterQualifier, IOQualifier ioQualifier)
		: parameterQualifier(parameterQualifier), ioQualifier(ioQualifier)
{
}

void TypePrefix::dump() const { dump(llvm::outs(), 0); }

void TypePrefix::dump(llvm::raw_ostream& os, size_t indents) const
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
