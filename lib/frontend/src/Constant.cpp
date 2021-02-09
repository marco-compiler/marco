#include <modelica/frontend/Constant.hpp>

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

void Constant::dump() const { dump(llvm::outs(), 0); }

void Constant::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "constant: " << *this << "\n";
}

SourcePosition Constant::getLocation() const
{
	return location;
}

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const Constant& obj)
{
	return stream << toString(obj);
}

class ConstantToStringVisitor {
	public:
	string operator()(const bool& value) { return value ? "true" : "false"; }
	string operator()(const int& value) { return to_string(value); }
	string operator()(const double& value) { return to_string(value); }
	string operator()(const string& value) { return value; }
};

std::string modelica::toString(const Constant& obj)
{
	return obj.visit(ConstantToStringVisitor());
}
