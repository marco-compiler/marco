#include <modelica/frontend/Equation.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Equation::Equation(SourcePosition location, Expression leftHand, Expression rightHand)
		: location(move(location)),
			leftHand(std::move(leftHand)),
			rightHand(std::move(rightHand))
{
}

void Equation::dump() const { dump(outs(), 0); }

void Equation::dump(raw_ostream& os, size_t indents) const
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
	this->leftHand = move(expression);
}

Expression& Equation::getRightHand() { return rightHand; }

const Expression& Equation::getRightHand() const { return rightHand; }

void Equation::setRightHand(Expression expression)
{
	this->rightHand = move(expression);
}
