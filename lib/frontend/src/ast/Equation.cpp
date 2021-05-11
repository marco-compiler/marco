#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace frontend;

Equation::Equation(SourcePosition location,
									 std::unique_ptr<Expression> lhs,
									 std::unique_ptr<Expression> rhs)
		: ASTNodeCRTP<Equation>(ASTNodeKind::EQUATION, std::move(location)),
			lhs(std::move(lhs)),
			rhs(std::move(rhs))
{
}

Equation::Equation(const Equation& other)
		: ASTNodeCRTP<Equation>(static_cast<const ASTNodeCRTP<Equation>&>(other)),
			lhs(other.lhs->cloneExpression()),
			rhs(other.rhs->cloneExpression())
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

namespace modelica::frontend
{
	void swap(Equation& first, Equation& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Equation>&>(first),
				 static_cast<impl::ASTNodeCRTP<Equation>&>(second));

		using std::swap;
		swap(first.lhs, second.lhs);
		swap(first.rhs, second.rhs);
	}
}

void Equation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "equation\n";
	lhs->dump(os, indents + 1);
	rhs->dump(os, indents + 1);
}

Expression* Equation::getLhsExpression()
{
	return lhs.get();
}

const Expression* Equation::getLhsExpression() const
{
	return lhs.get();
}

void Equation::setLhsExpression(Expression* expression)
{
	this->lhs = expression->cloneExpression();
}

Expression* Equation::getRhsExpression()
{
	return rhs.get();
}

const Expression* Equation::getRhsExpression() const
{
	return rhs.get();
}

void Equation::setRhsExpression(Expression* expression)
{
	this->rhs = expression->cloneExpression();
}
