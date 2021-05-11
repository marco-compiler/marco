#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Expression::Expression(ASTNodeKind kind, SourcePosition location, Type type)
		: ASTNodeCRTP<Expression>(kind, std::move(location)),
			type(std::move(type))
{
}

Expression::Expression(const Expression& other)
		: ASTNodeCRTP<Expression>(static_cast<const ASTNodeCRTP<Expression>&>(other)),
			type(other.type)
{
}

Expression::Expression(Expression&& other) = default;

Expression::~Expression() = default;

Expression& Expression::operator=(const Expression& other)
{
	if (this != &other)
	{
		static_cast<ASTNodeCRTP<Expression>&>(*this) = static_cast<const ASTNodeCRTP<Expression>&>(other);
		this->type = other.type;
	}

	return *this;
}

Expression& Expression::operator=(Expression&& other) = default;

namespace modelica::frontend
{
	void swap(Expression& first, Expression& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Expression>&>(first),
				 static_cast<impl::ASTNodeCRTP<Expression>&>(second));

		using std::swap;
		swap(first.type, second.type);
	}
}

bool Expression::operator!=(const Expression& rhs) const
{
	return !(rhs == *this);
}

Type& Expression::getType()
{
	return type;
}

const Type& Expression::getType() const
{
	return type;
}

void Expression::setType(Type tp)
{
	type = std::move(tp);
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Expression& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Expression& obj)
	{
		// TODO
		return "";
		//return obj.visit([](const auto& obj) { return toString(obj); });
	}
}
