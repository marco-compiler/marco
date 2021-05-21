#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Constant::Constant(SourceRange location, Type type, bool value)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			value(value)
{
}

Constant::Constant(SourceRange location, Type type, long value)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			value(value)
{
}

Constant::Constant(SourceRange location, Type type, double value)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			value(value)
{
}

Constant::Constant(SourceRange location, Type type, std::string value)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			value(value)
{
}

Constant::Constant(SourceRange location, Type type, int value)
		: Constant(std::move(location), std::move(type), static_cast<long>(value))
{
}

Constant::Constant(SourceRange location, Type type, float value)
		: Constant(std::move(location), std::move(type), static_cast<double >(value))
{
}

Constant::Constant(const Constant& other)
		: ASTNode(other),
			type(other.type),
			value(other.value)
{
}

Constant::Constant(Constant&& other) = default;

Constant::~Constant() = default;

Constant& Constant::operator=(const Constant& other)
{
	Constant result(other);
	swap(*this, result);
	return *this;
}

Constant& Constant::operator=(Constant&& other) = default;

namespace modelica::frontend
{
	void swap(Constant& first, Constant& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.type, second.type);
		swap(first.value, second.value);
	}
}

bool Constant::operator==(const Constant& other) const
{
	return type == other.type && value == other.value;
}

bool Constant::operator!=(const Constant& other) const
{
	return !(*this == other);
}

void Constant::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "constant: " << *this << "\n";
}

bool Constant::isLValue() const
{
	return false;
}

Type& Constant::getType()
{
	return type;
}

const Type& Constant::getType() const
{
	return type;
}

void Constant::setType(Type tp)
{
	type = std::move(tp);
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
		std::string operator()(const long& value) { return std::to_string(value); }
		std::string operator()(const double& value) { return std::to_string(value); }
		std::string operator()(const std::string& value) { return value; }
	};

	std::string toString(const Constant& obj)
	{
		return obj.visit(ConstantToStringVisitor());
	}
}
