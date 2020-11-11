#include <modelica/frontend/TypePrefix.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const ParameterQualifier& obj)
	{
		return stream << toString(obj);
	}

	string toString(ParameterQualifier qualifier)
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

	raw_ostream& operator<<(raw_ostream& stream, const IOQualifier& obj)
	{
		return stream << toString(obj);
	}

	string toString(IOQualifier qualifier)
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
}	 // namespace modelica

TypePrefix::TypePrefix(
		ParameterQualifier parameterQualifier, IOQualifier ioQualifier)
		: parameterQualifier(parameterQualifier), ioQualifier(ioQualifier)
{
}

void TypePrefix::dump() const { dump(outs(), 0); }

void TypePrefix::dump(raw_ostream& os, size_t indents) const
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

TypePrefix TypePrefix::empty()
{
	return TypePrefix(ParameterQualifier::none, IOQualifier::none);
}
