#include <modelica/frontend/AST.h>

using namespace modelica;

Equation::Equation(SourcePosition location, Expression leftHand, Expression rightHand)
		: location(std::move(location)),
			leftHand(std::move(leftHand)),
			rightHand(std::move(rightHand))
{
}

void Equation::dump() const { dump(llvm::outs(), 0); }

void Equation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "equation\n";
	leftHand.dump(os, indents + 1);
	rightHand.dump(os, indents + 1);
}

SourcePosition Equation::getLocation() const
{
	return location;
}

Expression& Equation::getLeftHand() { return leftHand; }

const Expression& Equation::getLeftHand() const { return leftHand; }

void Equation::setLeftHand(Expression expression)
{
	this->leftHand = std::move(expression);
}

Expression& Equation::getRightHand() { return rightHand; }

const Expression& Equation::getRightHand() const { return rightHand; }

void Equation::setRightHand(Expression expression)
{
	this->rightHand = std::move(expression);
}
