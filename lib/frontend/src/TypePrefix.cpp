#include <modelica/frontend/TypePrefix.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const ParameterQualifier& obj)
	{
		if (obj == ParameterQualifier::discrete)
			stream << "discrete";
		else if (obj == ParameterQualifier::parameter)
			stream << "parameter";
		else if (obj == ParameterQualifier::constant)
			stream << "constant";

		return stream;
	}

	raw_ostream& operator<<(raw_ostream& stream, const IOQualifier& obj)
	{
		if (obj == IOQualifier::input)
			stream << "discrete";
		else if (obj == IOQualifier::output)
			stream << "output";

		return stream;
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

bool TypePrefix::isParameter()
{
	return parameterQualifier == ParameterQualifier::parameter ||
				 parameterQualifier == ParameterQualifier::constant;
}
