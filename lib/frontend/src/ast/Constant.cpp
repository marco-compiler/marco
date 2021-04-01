#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace frontend;

Constant::Constant(SourcePosition location, bool val)
		: location(std::move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, int val)
		: location(std::move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, float val)
		: location(std::move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, double val)
		: location(std::move(location)),
			content(static_cast<float>(val))
{
}

Constant::Constant(SourcePosition location, char val)
		: location(std::move(location)),
			content(val)
{
}

Constant::Constant(SourcePosition location, std::string val)
		: location(std::move(location)),
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

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Constant& obj)
	{
		return stream << toString(obj);
	}

	class ConstantToStringVisitor {
		public:
		std::string operator()(const bool& value) { return value ? "true" : "false"; }
		std::string operator()(const int& value) { return std::to_string(value); }
		std::string operator()(const double& value) { return std::to_string(value); }
		std::string operator()(const std::string& value) { return value; }
	};

	std::string toString(const Constant& obj)
	{
		return obj.visit(ConstantToStringVisitor());
	}
}
