#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

using namespace modelica::codegen::model;

Equation::Equation(Expression left,
										 Expression right,
										 std::string templateName,
										 MultiDimInterval inds,
										 bool isForward,
										 std::optional<EquationPath> path)
		: body(std::make_shared<EquationTemplate>(left, right, templateName)),
			inductions(std::move(inds)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward),
			matchedExpPath(std::move(path))
{
}

Equation::Equation(std::shared_ptr<EquationTemplate> templ,
										 MultiDimInterval interval,
										 bool isForward)
		: body(std::move(templ)),
			inductions(std::move(interval)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward)
{
	if (!isForCycle)
		inductions = { { 0, 1 } };
}

Expression& Equation::lhs()
{
	return body->lhs();
}

const Expression& Equation::lhs() const
{
	return body->lhs();
}

Expression& Equation::rhs()
{
	return body->rhs();
}

const Expression& Equation::rhs() const
{
	return body->rhs();
}

std::shared_ptr<EquationTemplate>& Equation::getTemplate()
{
	return body;
}

const std::shared_ptr<EquationTemplate>& Equation::getTemplate() const
{
	return body;
}

const modelica::MultiDimInterval& Equation::getInductions() const
{
	return inductions;
}

void Equation::setInductionVars(MultiDimInterval inds)
{
	isForCycle = !inds.empty();

	if (isForCycle)
		inductions = std::move(inds);
	else
		inductions = { { 0, 1 } };
}

bool Equation::isForEquation() const
{
	return isForCycle;
}

size_t Equation::dimensions() const
{
	return isForCycle ? inductions.dimensions() : 0;
}

bool Equation::isForward() const
{
	return isForwardDirection;
}

void Equation::setForward(bool isForward)
{
	isForwardDirection = isForward;
}

bool Equation::isMatched() const
{
	return matchedExpPath.has_value();
}

Expression& Equation::getMatchedExp()
{
	assert(isMatched());
	return reachExp(matchedExpPath.value());
}

const Expression& Equation::getMatchedExp() const
{
	assert(isMatched());
	return reachExp(matchedExpPath.value());
}

void Equation::setMatchedExp(EquationPath path)
{
	assert(reachExp(path).isA<Reference>());
	matchedExpPath = path;
}

AccessToVar Equation::getDeterminedVariable() const
{
	assert(isMatched());
	return AccessToVar::fromExp(getMatchedExp());
}

Equation Equation::clone(std::string newName) const
{
	Equation clone = *this;
	clone.body = std::make_shared<EquationTemplate>(*body);
	clone.getTemplate()->setName(std::move(newName));
	return clone;
}

EquationTemplate::EquationTemplate(Expression left, Expression right, std::string name)
		: left(std::make_shared<Expression>(left)),
			right(std::make_shared<Expression>(right)),
			name(std::move(name))
{
}

Expression& EquationTemplate::lhs()
{
	return *left;
}

const Expression& EquationTemplate::lhs() const
{
	return *left;
}

Expression& EquationTemplate::rhs()
{
	return *right;
}

const Expression& EquationTemplate::rhs() const
{
	return *right;
}

std::string& EquationTemplate::getName()
{
	return name;
}

const std::string& EquationTemplate::getName() const
{
	return name;
}

void EquationTemplate::setName(std::string newName)
{
	name = newName;
}

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
		e = &e->getChild(i);

	return *e;
}

const Expression& EquationPath::reach(const Expression& exp) const
{
	const Expression* e = &exp;

	for (auto i : path)
		e = &e->getChild(i);

	return *e;
}