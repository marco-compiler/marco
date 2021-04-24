#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/Path.h>
#include <modelica/utils/ScopeGuard.hpp>

using namespace llvm;
using namespace std;
using namespace modelica::codegen::model;

ReferenceMatcher::ReferenceMatcher() = default;
ReferenceMatcher::ReferenceMatcher(Equation eq)
{
	visit(eq);
}

ExpressionPath& ReferenceMatcher::operator[](size_t index)
{
	return at(index);
}

const ExpressionPath& ReferenceMatcher::operator[](size_t index) const
{
	return at(index);
}

ReferenceMatcher::iterator ReferenceMatcher::begin()
{
	return vars.begin();
}

ReferenceMatcher::const_iterator ReferenceMatcher::begin() const
{
	return vars.begin();
}

ReferenceMatcher::iterator ReferenceMatcher::end()
{
	return vars.end();
}

ReferenceMatcher::const_iterator ReferenceMatcher::end() const
{
	return vars.end();
}

size_t ReferenceMatcher::size() const
{
	return vars.size();
}

ExpressionPath& ReferenceMatcher::at(size_t index)
{
	return vars[index];
}

const ExpressionPath& ReferenceMatcher::at(size_t index) const
{
	return vars[index];
}

Expression ReferenceMatcher::getExp(size_t index) const
{
	return at(index).getExpression();
}

void ReferenceMatcher::visit(Equation equation, bool ignoreMatched)
{
	assert(!ignoreMatched || equation.isMatched());
	visit(equation.lhs(), true, 0);
	visit(equation.rhs(), false, 0);

	if (!ignoreMatched)
		return;

	auto match = equation.getMatchedExp();

	vars.erase(
			remove_if(
					vars,
					[match](const ExpressionPath& path) { return path.getExpression() == match; }),
			vars.end());
}

void ReferenceMatcher::visit(Expression exp, bool isLeft, size_t index)
{
	if (exp.isReferenceAccess())
	{
		vars.emplace_back(exp, currentPath, isLeft);
		return;
	}

	if (mlir::isa<CallOp>(exp.getOp()))
		return;

	for (size_t i = 0; i < exp.childrenCount(); ++i)
	{
		currentPath.push_back(i);
		auto g = makeGuard(std::bind(&ReferenceMatcher::removeBack, this));
		visit(exp.getChild(i), isLeft, i);
	}
}

void ReferenceMatcher::removeBack()
{
	currentPath.erase(currentPath.end() - 1, currentPath.end());
}
