#include "modelica/model/ModExp.hpp"

#include "modelica/model/ModConst.hpp"

using namespace modelica;

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

	if (isUnary() && !getLeftHand().isConstant())
		return false;

	if (isBinary())
	{
		if (!getLeftHand().isConstant())
			return false;
		if (!getRightHand().isConstant())
			return false;
	}

	if (isTernary())
	{
		if (!getLeftHand().isConstant())
			return false;
		if (!getRightHand().isConstant())
			return false;
		if (!getCondition().isConstant())
			return false;
	}

	switch (getKind())
	{
		case ModExpKind::zero:
			return false;
		case ModExpKind::negate:
			getConstant().negateAll();
			return true;
		case ModExpKind::induction:
			return false;
		case ModExpKind::add:
			*this = ModExp(ModConst::sum(
					getLeftHand().getConstant(), getRightHand().getConstant()));
			return true;
		case ModExpKind::sub:
		case ModExpKind::at:
		case ModExpKind::mult:
		case ModExpKind::divide:
		case ModExpKind::greaterThan:
		case ModExpKind::greaterEqual:
		case ModExpKind::equal:
		case ModExpKind::different:
		case ModExpKind::less:
		case ModExpKind::lessEqual:
		case ModExpKind::elevation:
		case ModExpKind::module:
		case ModExpKind::conditional:
			return false;
	}

	assert(false && "unreachable");
	return false;
}
