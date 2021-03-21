#include <modelica/frontend/ast/Modification.h>

using namespace modelica;

Modification::Modification(ClassModification classModification)
		: classModification(std::make_shared<ClassModification>(classModification)),
			expression(llvm::None)
{
}

Modification::Modification(ClassModification classModification, Expression expression)
		: classModification(std::make_shared<ClassModification>(classModification)),
			expression(std::make_shared<Expression>(expression))
{
}

Modification::Modification(Expression expression)
		: classModification(llvm::None),
			expression(std::make_shared<Expression>(expression))
{
}

bool Modification::hasClassModification() const
{
	return classModification.hasValue();
}

ClassModification& Modification::getClassModification()
{
	assert(hasClassModification());
	return **classModification;
}

const ClassModification& Modification::getClassModification() const
{
	assert(hasClassModification());
	return **classModification;
}

bool Modification::hasExpression() const
{
	return expression.hasValue();
}

Expression& Modification::getExpression()
{
	assert(hasExpression());
	return **expression;
}

const Expression& Modification::getExpression() const
{
	assert(hasExpression());
	return **expression;
}

ClassModification::ClassModification(llvm::ArrayRef<Argument> arguments)
{
	for (const auto& arg : arguments)
		this->arguments.push_back(std::make_shared<Argument>(arg));
}

ClassModification::iterator<Argument> ClassModification::begin()
{
	return arguments.begin();
}

ClassModification::const_iterator<Argument> ClassModification::begin() const
{
	return arguments.begin();
}

ClassModification::iterator<Argument> ClassModification::end()
{
	return arguments.end();
}

ClassModification::const_iterator<Argument> ClassModification::end() const
{
	return arguments.end();
}

Argument::Argument(ElementModification content)
		: content(std::make_shared<ElementModification>(content))
{
}

Argument::Argument(ElementRedeclaration content)
		: content(std::make_shared<ElementRedeclaration>(content))
{
}

Argument::Argument(ElementReplaceable content)
		: content(std::make_shared<ElementReplaceable>(content))
{
}

ElementModification::ElementModification(bool each, bool final, std::string name, Modification modification)
		: each(each),
			final(final),
			name(std::move(name)),
			modification(std::make_shared<Modification>(modification))
{
}

ElementModification::ElementModification(bool each, bool final, std::string name)
		: each(each),
			final(final),
			name(std::move(name))
{
}

bool ElementModification::hasEachProperty() const
{
	return each;
}

bool ElementModification::hasFinalProperty() const
{
	return final;
}

std::string& ElementModification::getName()
{
	return name;
}

const std::string& ElementModification::getName() const
{
	return name;
}

bool ElementModification::hasModification() const
{
	return modification.hasValue();
}

Modification& ElementModification::getModification()
{
	assert(hasModification());
	return **modification;
}

const Modification& ElementModification::getModification() const
{
	assert(hasModification());
	return **modification;
}
