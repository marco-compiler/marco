#include "modelica/frontend/Statement.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

Statement::Statement(Expression destination, Expression expression)
		: Statement(vector({ destination }), expression)
{
}

Statement::Statement(ArrayRef<Expression> destinations, Expression expression)
		: destinations(
					iterator_range<ArrayRef<Expression>::iterator>(move(destinations))),
			expression(move(expression))
{
}

void Statement::dump(raw_ostream& OS, size_t indents) const
{
	OS.indent(indents);
	OS << "destinations: "
		 << "\n";

	for (const auto& destination : destinations)
		destination.dump(OS, indents + 1);

	OS.indent(indents);
	OS << "expression: "
		 << "\n";
	expression.dump(OS, indents + 1);
}

SmallVectorImpl<Expression>& Statement::getDestinations()
{
	return destinations;
}

Expression& Statement::getExpression() { return expression; }

const SmallVectorImpl<Expression>& Statement::getDestinations() const
{
	return destinations;
}

const Expression& Statement::getExpression() const { return expression; }
