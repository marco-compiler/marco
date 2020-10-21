#include <modelica/frontend/Equation.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

Equation::Equation(Expression leftHand, Expression rightHand)
		: leftHand(std::move(leftHand)), rightHand(std::move(rightHand))
{
}

void Equation::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "equation\n";
	leftHand.dump(os, indents + 1);
	rightHand.dump(os, indents + 1);
}

Expression& Equation::getLeftHand() { return leftHand; }

Expression& Equation::getRightHand() { return rightHand; }

const Expression& Equation::getLeftHand() const { return leftHand; }

const Expression& Equation::getRightHand() const { return rightHand; }
