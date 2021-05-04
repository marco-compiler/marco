#include "modelica/model/SymbolicDifferentiation.hpp"

#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"

using namespace modelica;
using namespace std;

template<ModExpKind kind>
static ModExp differentiateOp(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	assert(false && "Unreachable");
}

template<>
ModExp differentiateOp<ModExpKind::at>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	assert(exp.isReferenceAccess());

	// If the variable is scalar and the expression is not, return 0.
	if (var.size() == 1)
		return ModConst(0.0);

	assert(VectorAccess::isCanonical(ind));

	// If the expression is a variable different from the given one, return 0.
	if (exp.getReferredVectorAccess() != var.getName())
		return ModConst(0.0);

	// If the expression have a different index from the given one, return 0.
	if (AccessToVar::fromExp(exp).getAccess() !=
			AccessToVar::fromExp(ind).getAccess())
		return ModConst(0.0);

	// Otherwise, return 1.
	return ModConst(1.0);
}

template<>
ModExp differentiateOp<ModExpKind::negate>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If the expression is constant, return 0.
	if (exp.getLeftHand().isConstant())
		return ModConst(0.0);

	// Otherwise, return the negated derivative of the espression.
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);

	ModExp result = ModExp::negate(move(leftDerivative));
	result.tryFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::add>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the other side.
	if (exp.getLeftHand().isConstant())
		return differentiate(exp.getRightHand(), var, ind);
	if (exp.getRightHand().isConstant())
		return differentiate(exp.getLeftHand(), var, ind);

	// Otherwise, return der(f(x)) + der(g(x)).
	ModExp leftHand = differentiate(exp.getLeftHand(), var, ind);
	ModExp rightHand = differentiate(exp.getRightHand(), var, ind);

	ModExp result = ModExp::add(move(leftHand), move(rightHand));
	result.tryFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::sub>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the other side.
	if (exp.getRightHand().isConstant())
		return differentiate(exp.getLeftHand(), var, ind);

	ModExp rightHand = differentiate(exp.getRightHand(), var, ind);

	if (exp.getLeftHand().isConstant())
	{
		ModExp result = ModExp::negate(rightHand);
		result.tryFoldConstant();
		return result;
	}

	// Otherwise, return der(f(x)) - der(g(x)).
	ModExp leftHand = differentiate(exp.getLeftHand(), var, ind);

	ModExp result = ModExp::subtract(move(leftHand), move(rightHand));
	result.tryFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::mult>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the otherside multiplied by
	// the constant.
	if (exp.getLeftHand().isConstant())
	{
		ModExp rightDerivative = differentiate(exp.getRightHand(), var, ind);
		ModExp result = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
		result.tryFoldConstant();
		return result;
	}
	if (exp.getRightHand().isConstant())
	{
		ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);
		ModExp result = ModExp::multiply(exp.getRightHand(), move(leftDerivative));
		result.tryFoldConstant();
		return result;
	}

	// Otherwise, return f(x)*der(g(x)) + g(x)*der(f(x)).
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);
	ModExp rightDerivative = differentiate(exp.getRightHand(), var, ind);

	ModExp leftHand = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
	ModExp rightHand = ModExp::multiply(exp.getRightHand(), move(leftDerivative));

	ModExp result = ModExp::add(move(leftHand), move(rightHand));
	result.tryRecursiveFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::divide>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If the divisor is a constant, return der(f(x)) / g(x)
	if (exp.getRightHand().isConstant())
	{
		ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);
		ModExp result = ModExp::divide(move(leftDerivative), exp.getRightHand());
		result.tryFoldConstant();
		return result;
	}

	// If the dividend is constant, return -f(x)*der(g(x)) / (g(x)*g(x))
	if (exp.getLeftHand().isConstant())
	{
		ModExp rightDerivative = differentiate(exp.getRightHand(), var, ind);
		ModExp dividend = ModExp::negate(
				ModExp::multiply(exp.getLeftHand(), move(rightDerivative)));
		ModExp divisor = ModExp::multiply(exp.getRightHand(), exp.getRightHand());

		ModExp result = ModExp::divide(move(dividend), move(divisor));
		result.tryRecursiveFoldConstant();
		return result;
	}

	// Otherwise, return (g(x)*der(f(x)) - f(x)*der(g(x))) / (g(x) * g(x)).
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);
	ModExp rightDerivative = differentiate(exp.getRightHand(), var, ind);

	ModExp lDividend = ModExp::multiply(exp.getRightHand(), move(leftDerivative));
	ModExp rDividend = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));

	ModExp dividend = ModExp::subtract(move(lDividend), move(rDividend));
	ModExp divisor = ModExp::multiply(exp.getRightHand(), exp.getRightHand());

	ModExp result = ModExp::divide(move(dividend), move(divisor));
	result.tryRecursiveFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::elevation>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	assert(exp.getRightHand().isConstant());

	ModConst exponent = exp.getRightHand().getConstant().as<double>();

	// If the base is constant, return 0.
	if (exp.getLeftHand().isConstant())
		return ModConst(0.0);

	// If the exponent is 1, return the derivative of the base.
	if (exponent == ModConst(1.0))
		return differentiate(exp.getLeftHand(), var, ind);

	ModExp leftDerivative = differentiate(exp.getLeftHand(), var, ind);

	// If the exponent is 2, return two times the derivative of the base.
	if (exponent == ModConst(2.0))
	{
		ModExp right = ModExp::multiply(exp.getLeftHand(), move(leftDerivative));

		ModExp result = ModExp::multiply(ModConst(2.0), move(right));
		result.tryRecursiveFoldConstant();
		return result;
	}

	// Otherwise, return  (c * der(f(x))) * exp(f(x), (c-1)).
	ModExp elevationResidual = ModExp::elevate(
			exp.getLeftHand(), ModConst::sub(exponent, ModConst(1.0)));
	ModExp left = ModExp::multiply(exponent, move(leftDerivative));

	ModExp result = ModExp::multiply(move(left), move(elevationResidual));
	result.tryRecursiveFoldConstant();
	return result;
}

template<>
ModExp differentiateOp<ModExpKind::induction>(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	return ModConst(0.0);
}

ModExp modelica::differentiate(
		const ModExp& exp, const ModVariable& var, const ModExp& ind)
{
	// If the expression is a constant, return 0.
	if (exp.isConstant())
		return ModConst(0.0);

	// If the expression is a variable different from the given one, return 0.
	if (exp.isReference() && exp.getReference() != var.getName())
		return ModConst(0.0);

	// If the expression is the same variable as the given one, return 1.
	if (exp.isReference() && exp.getReference() == var.getName())
		return ModConst(1.0);

	assert(exp.isOperation());

	// Otherwise, differentiate the expression depending on the operation.
	switch (exp.getKind())
	{
		case ModExpKind::at:
			return differentiateOp<ModExpKind::at>(exp, var, ind);
		case ModExpKind::negate:
			return differentiateOp<ModExpKind::negate>(exp, var, ind);
		case ModExpKind::add:
			return differentiateOp<ModExpKind::add>(exp, var, ind);
		case ModExpKind::sub:
			return differentiateOp<ModExpKind::sub>(exp, var, ind);
		case ModExpKind::mult:
			return differentiateOp<ModExpKind::mult>(exp, var, ind);
		case ModExpKind::divide:
			return differentiateOp<ModExpKind::divide>(exp, var, ind);
		case ModExpKind::elevation:
			return differentiateOp<ModExpKind::elevation>(exp, var, ind);
		case ModExpKind::induction:
			return differentiateOp<ModExpKind::induction>(exp, var, ind);
		default:
			assert(false && "Not differentiable expression");
	}

	assert(false && "Unreachable");
}

ModEquation modelica::differentiate(
		const ModEquation& eq, const ModVariable& var, const ModExp& ind)
{
	ModExp leftHand = differentiate(eq.getLeft(), var, ind);
	ModExp rightHand = differentiate(eq.getRight(), var, ind);

	return ModEquation(move(leftHand), move(rightHand));
}
