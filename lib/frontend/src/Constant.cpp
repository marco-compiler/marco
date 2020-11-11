#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Type.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Constant::Constant(int val): content(val) {}
Constant::Constant(char val): content(val) {}
Constant::Constant(float val): content(val) {}
Constant::Constant(bool val): content(val) {}
Constant::Constant(double val): content(static_cast<float>(val)) {}
Constant::Constant(string val): content(move(val)) {}

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

	if (isA<BuiltinType::Integer>())
		os << get<BuiltinType::Integer>();
	else if (isA<BuiltinType::Float>())
		os << get<BuiltinType::Float>();
	else if (isA<BuiltinType::Boolean>())
		os << (get<BuiltinType::Boolean>() ? "true" : "false");
	else if (isA<BuiltinType::String>())
		os << get<BuiltinType::String>();
	else
		assert(false && "Unreachable");

	os << "\n";
}
