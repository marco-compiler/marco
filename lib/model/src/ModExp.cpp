#include "modelica/model/ModExp.hpp"

#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "modelica/model/ModConst.hpp"

using namespace modelica;
using namespace std;

static std::string exprKindToString(ModExpKind kind)
{
	switch (kind)
	{
		case (ModExpKind::zero):
			return "0";
		case (ModExpKind::negate):
			return "!";
		case (ModExpKind::add):
			return "+";
		case (ModExpKind::sub):
			return "-";
		case (ModExpKind::mult):
			return "*";
		case (ModExpKind::at):
			return "at";
		case (ModExpKind::divide):
			return "/";
		case (ModExpKind::greaterThan):
			return ">";
		case (ModExpKind::greaterEqual):
			return ">=";
		case (ModExpKind::equal):
			return "==";
		case (ModExpKind::different):
			return "!=";
		case (ModExpKind::less):
			return "<";
		case (ModExpKind::lessEqual):
			return "<=";
		case (ModExpKind::elevation):
			return "^";
		case (ModExpKind::module):
			return "%";
		case (ModExpKind::conditional):
			return "?";
		case (ModExpKind::induction):
			return "ind";
	}
	assert(false);	// NOLINT
	return "UNREACHABLE";
}

ModType ModExp::Operation::getOperationReturnType() const
{
	switch (kind)
	{
		case ModExpKind::zero:
			return ModType(BultinModTypes::BOOL);
		case ModExpKind::induction:
			return ModType(BultinModTypes::INT);
		case ModExpKind::at:
			return leftHandExpression->getModType().sclidedType();
		case ModExpKind::negate:
		case ModExpKind::add:
		case ModExpKind::sub:
		case ModExpKind::mult:
		case ModExpKind::divide:
		case ModExpKind::elevation:
		case ModExpKind::module:
		case ModExpKind::conditional:
			return leftHandExpression->getModType();
		case ModExpKind::greaterThan:
		case ModExpKind::greaterEqual:
		case ModExpKind::equal:
		case ModExpKind::different:
		case ModExpKind::less:
		case ModExpKind::lessEqual:
			return leftHandExpression->getModType().as(BultinModTypes::BOOL);
	}
	assert(false && "Unreachable");	 // NOLINT
	return ModType(BultinModTypes::BOOL);
}

static void dumpOperation(const ModExp& exp, llvm::raw_ostream& OS)
{
	OS << '(';
	OS << exprKindToString(exp.getKind());
	OS << ' ';

	if (exp.getArity() >= 1)
		exp.getLeftHand().dump(OS);
	if (exp.getArity() >= 2)
	{
		OS << ", ";
		exp.getRightHand().dump(OS);
	}
	if (exp.getArity() == 3)
	{
		exp.getCondition().dump(OS);
		OS << ", ";
	}
	OS << ')';
}

bool ModExp::isReferenceAccess() const
{
	if (isReference())
		return true;

	if (isOperation())
		if (getOperation().getKind() == ModExpKind::at)
			return getLeftHand().isReferenceAccess();
	return false;
}

void ModExp::dump(llvm::raw_ostream& OS) const
{
	getModType().dump(OS);
	if (isConstant())
	{
		getConstant().dump(OS);
		return;
	}
	if (isReference())
	{
		OS << getReference();
		return;
	}

	if (isOperation())
	{
		dumpOperation(*this, OS);
		return;
	}
	if (isCall())
	{
		getCall().dump(OS);
		return;
	}
	assert(false && "Unrechable");	// NOLINT
}

bool ModExp::tryFoldConstant()
{
	if (!isOperation())
		return false;

	if (isUnary())
	{
		if (!getLeftHand().isConstant() || !isOperation<ModExpKind::negate>())
			return false;

		getLeftHand().getConstant().negateAll();
		*this = ModExp(move(getLeftHand().getConstant()), getModType());
		return true;
	}

	if (isBinary())
		if (!all_of(range(), [](const auto& exp) { return exp.isConstant(); }))
			return false;

	if (isTernary() && !getCondition().isConstant())
		return false;

	const auto& lConst = getLeftHand().getConstant();
	const auto& rConst = getRightHand().getConstant();

	switch (getKind())
	{
		case ModExpKind::zero:
		case ModExpKind::negate:
		case ModExpKind::induction:
		case ModExpKind::at:
			return false;
		case ModExpKind::add:
			*this = ModExp(ModConst::sum(lConst, rConst), getModType());
			return true;
		case ModExpKind::sub:
			*this = ModExp(ModConst::sub(lConst, rConst), getModType());
			return true;
		case ModExpKind::mult:
			*this = ModExp(ModConst::mult(lConst, rConst), getModType());
			return true;
		case ModExpKind::divide:
			*this = ModExp(ModConst::divide(lConst, rConst), getModType());
			return true;
		case ModExpKind::greaterThan:
			*this = ModExp(ModConst::greaterThan(lConst, rConst), getModType());
			return true;
		case ModExpKind::greaterEqual:
			*this = ModExp(ModConst::greaterEqual(lConst, rConst), getModType());
			return true;
		case ModExpKind::equal:
			*this = ModExp(ModConst::equal(lConst, rConst), getModType());
			return true;
		case ModExpKind::different:
			*this = ModExp(ModConst::different(lConst, rConst), getModType());
			return true;
		case ModExpKind::less:
			*this = ModExp(ModConst::lessThan(lConst, rConst), getModType());
			return true;
		case ModExpKind::lessEqual:
			*this = ModExp(ModConst::lessEqual(lConst, rConst), getModType());
			return true;
		case ModExpKind::elevation:
			*this = ModExp(ModConst::elevate(lConst, rConst), getModType());
			return true;
		case ModExpKind::module:
			*this = ModExp(ModConst::module(lConst, rConst), getModType());
			return true;
		case ModExpKind::conditional:
			if (getCondition().getConstant().get<bool>(0))
			{
				auto newVal = move(getLeftHand());
				*this = move(newVal);
			}
			else
			{
				auto newVal = move(getRightHand());
				*this = move(newVal);
			}
			return true;
	}

	assert(false && "unreachable");
	return false;
}

const string& ModExp::getReferredVectorAccesss() const
{
	assert(isReferenceAccess());
	if (isReference())
		return getReference();

	const auto* exp = this;

	while (exp->isOperation<ModExpKind::at>())
		exp = &exp->getLeftHand();

	return exp->getReference();
}
