#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Constant::Constant(SourcePosition location, Type type, bool value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(SourcePosition location, Type type, int value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(SourcePosition location, Type type, float value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(SourcePosition location, Type type, double value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(SourcePosition location, Type type, char value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(SourcePosition location, Type type, std::string value)
		: ExpressionCRTP<Constant>(
					ASTNodeKind::EXPRESSION_CONSTANT, std::move(location), std::move(type)),
			value(value)
{
}

Constant::Constant(const Constant& other)
		: ExpressionCRTP<Constant>(static_cast<ExpressionCRTP&>(*this)),
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
		swap(static_cast<impl::ExpressionCRTP<Constant>&>(first),
				 static_cast<impl::ExpressionCRTP<Constant>&>(second));

		using std::swap;
		swap(first.value, second.value);
	}
}

bool Constant::operator==(const Constant& other) const
{
	return value == other.value;
}

bool Constant::operator!=(const Constant& other) const
{
	return value != other.value;
}

void Constant::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "constant: " << *this << "\n";
}

bool Constant::isLValue() const
{
	return false;
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
