#include <marco/frontend/AST.h>

using namespace marco::frontend;

Equation::Equation(SourceRange location,
									 std::unique_ptr<Expression> lhs,
									 std::unique_ptr<Expression> rhs)
		: ASTNode(std::move(location)),
			lhs(std::move(lhs)),
			rhs(std::move(rhs))
{
}

Equation::Equation(const Equation& other)
		: ASTNode(other),
			lhs(other.lhs->clone()),
			rhs(other.rhs->clone())
{
}

Equation::Equation(Equation&& other) = default;

Equation::~Equation() = default;

Equation& Equation::operator=(const Equation& other)
{
	Equation result(other);
	swap(*this, result);
	return *this;
}

Equation& Equation::operator=(Equation&& other) = default;

namespace marco::frontend
{
	void swap(Equation& first, Equation& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.lhs, second.lhs);
		swap(first.rhs, second.rhs);
	}
}

void Equation::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "equation\n";
	lhs->print(os, indents + 1);
	rhs->print(os, indents + 1);
}

Expression* Equation::getLhsExpression()
{
	return lhs.get();
}

const Expression* Equation::getLhsExpression() const
{
	return lhs.get();
}

void Equation::setLhsExpression(std::unique_ptr<Expression> expression)
{
	this->lhs = std::move(expression);
}

Expression* Equation::getRhsExpression()
{
	return rhs.get();
}

const Expression* Equation::getRhsExpression() const
{
	return rhs.get();
}

void Equation::setRhsExpression(std::unique_ptr<Expression> expression)
{
	this->rhs = std::move(expression);
}
