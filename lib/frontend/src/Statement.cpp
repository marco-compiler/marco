#include <modelica/frontend/Statement.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Statement::Statement(Expression destination, Expression expression)
		: destination(move(destination)), expression(move(expression))
{
}

Statement::Statement(
		initializer_list<Expression> destinations, Expression expression)
		: destination(Tuple(move(destinations))), expression(move(expression))
{
}

void Statement::dump() const { dump(outs(), 0); }

void Statement::dump(raw_ostream& os, size_t indents) const
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

vector<Expression*> Statement::getDestinations()
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

Expression& Statement::getExpression() { return expression; }

const Expression& Statement::getExpression() const { return expression; }
