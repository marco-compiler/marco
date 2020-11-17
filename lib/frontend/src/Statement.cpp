#include <modelica/frontend/Statement.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

AssignmentStatement::AssignmentStatement(
		Expression destination, Expression expression)
		: destination(move(destination)), expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		Tuple destinations, Expression expression)
		: destination(move(destinations)), expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		initializer_list<Expression> destinations, Expression expression)
		: destination(Tuple(move(destinations))), expression(move(expression))
{
}

void AssignmentStatement::dump() const { dump(outs(), 0); }

void AssignmentStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "destinations:\n";

	if (destinationIsA<Expression>())
		get<Expression>(destination).dump(os, indents + 1);
	else
		get<Tuple>(destination).dump(os, indents + 1);

	os.indent(indents);
	os << "assigned expression:\n";
	expression.dump(os, indents + 1);
}

vector<Expression*> AssignmentStatement::getDestinations()
{
	vector<Expression*> destinations;

	if (destinationIsA<Expression>())
		destinations.push_back(&getDestination<Expression>());
	else
	{
		for (auto& exp : getDestination<Tuple>())
			destinations.push_back(&*exp);
	}

	return destinations;
}

void AssignmentStatement::setDestination(Expression dest)
{
	destination = move(dest);
}

void AssignmentStatement::setDestination(Tuple dest)
{
	destination = move(dest);
}

Expression& AssignmentStatement::getExpression() { return expression; }

const Expression& AssignmentStatement::getExpression() const
{
	return expression;
}

Statement::Statement(AssignmentStatement statement): content(statement) {}

Statement::Statement(ForStatement statement): content(statement) {}