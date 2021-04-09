#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Path.h>

using namespace modelica::codegen;
using namespace model;

EquationPath::EquationPath(llvm::SmallVector<size_t, 3> path, bool left)
		: path(std::move(path)), left(left)
{
}

EquationPath::const_iterator EquationPath::begin() const
{
	return path.begin();
}

EquationPath::const_iterator EquationPath::end() const
{
	return path.end();
}

size_t EquationPath::depth() const
{
	return path.size();
}

bool EquationPath::isOnEquationLeftHand() const
{
	return left;
}

Expression EquationPath::reach(Expression exp) const
{
	Expression e = exp;

	for (auto i : path)
		e = e.getChild(i);

	return e;
}

ExpressionPath::ExpressionPath(Expression exp, llvm::SmallVector<size_t, 3> path, bool left)
		: path(std::move(path), left), exp(std::make_shared<Expression>(exp))
{
}

ExpressionPath::ExpressionPath(Expression exp, EquationPath path)
		: path(std::move(path)), exp(std::make_shared<Expression>(exp))
{
}

EquationPath::const_iterator ExpressionPath::begin() const
{
	return path.begin();
}

EquationPath::const_iterator ExpressionPath::end() const
{
	return path.end();
}

size_t ExpressionPath::depth() const
{
	return path.depth();
}

Expression ExpressionPath::getExp() const
{
	return *exp;
}

const EquationPath& ExpressionPath::getEqPath() const
{
	return path;
}

bool ExpressionPath::isOnEquationLeftHand() const
{
	return path.isOnEquationLeftHand();
}

Expression ExpressionPath::reach(Expression exp) const
{
	return path.reach(exp);
}
