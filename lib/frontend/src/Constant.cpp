#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Type.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Constant::Constant(SourcePosition location, bool val)
		: location(move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, int val)
		: location(move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, float val)
		: location(move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, double val)
		: location(move(location)),
			content(static_cast<float>(val))
{
}

Constant::Constant(SourcePosition location, char val)
		: location(move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, string val)
		: location(move(location)),
			content(move(val))
{
}

bool Constant::operator==(const Constant& other) const
{
	return content == other.content;
}

bool Constant::operator!=(const Constant& other) const
{
	return !(*this == other);
}

void Constant::dump() const { dump(outs(), 0); }

void Constant::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "constant: ";

	if (isA<BuiltInType::Integer>())
		os << get<BuiltInType::Integer>();
	else if (isA<BuiltInType::Float>())
		os << get<BuiltInType::Float>();
	else if (isA<BuiltInType::Boolean>())
		os << (get<BuiltInType::Boolean>() ? "true" : "false");
	else if (isA<BuiltInType::String>())
		os << get<BuiltInType::String>();
	else
		assert(false && "Unreachable");

	os << "\n";
}

SourcePosition Constant::getLocation() const
{
	return location;
}
