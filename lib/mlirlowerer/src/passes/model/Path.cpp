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

Expression& EquationPath::reach(Expression& exp) const
{
	Expression* e = &exp;

	for (auto i : path)
		e = e->getChild(i).get();

	return *e;
}

const Expression& EquationPath::reach(const Expression& exp) const
{
	const Expression* e = &exp;

	for (auto i : path)
		e = e->getChild(i).get();

	return *e;
}

ExpressionPath::ExpressionPath(const Expression& exp, llvm::SmallVector<size_t, 3> path, bool left)
		: path(std::move(path), left), exp(&exp)
{
}

ExpressionPath::ExpressionPath(const Expression& exp, EquationPath path)
		: path(std::move(path)), exp(&exp)
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

const Expression& ExpressionPath::getExp() const
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

Expression& ExpressionPath::reach(Expression& exp) const
{
	return path.reach(exp);
}

const Expression& ExpressionPath::reach(const Expression& exp) const
{
	return path.reach(exp);
}
