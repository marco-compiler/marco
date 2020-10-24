#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Type.hpp>

using namespace modelica;
using namespace llvm;
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

	if (isA<int>())
		os << get<int>();
	else if (isA<float>())
		os << get<float>();
	else if (isA<bool>())
		os << (get<bool>() ? "true" : "false");
	else if (isA<char>())
		os << get<char>();
	else if (isA<std::string>())
		os << get<std::string>();
	else
		assert(false && "unreachable");
}
