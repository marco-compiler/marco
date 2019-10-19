#include "modelica/model/ModExp.hpp"

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
	OS << "(";
	OS << exprKindToString(exp.getKind());
	OS << " ";
}

class ModExpDumper
{
	public:
	ModExpDumper(llvm::raw_ostream& OS): OS(OS) {}

	void visit(const ModExp& exp)
	{
		if (exp.isConstant<int>())
		{
			dumpConstant(exp.getConstant<int>(), OS);
			return;
		}
		if (exp.isConstant<float>())
		{
			dumpConstant(exp.getConstant<float>(), OS);
			return;
		}
		if (exp.isConstant<bool>())
		{
			dumpConstant(exp.getConstant<bool>(), OS);
			return;
		}
		if (exp.isReference())
		{
			OS << exp.getReference();
			return;
		}

		if (exp.isOperation())
		{
			dumpOperation(exp, OS);
			return;
		}
		if (exp.isCall())
		{
			exp.getCall().dump(OS);
			return;
		}
		assert(false && "Unrechable");	// NOLINT
	}

	void afterVisit(const ModExp& exp)
	{
		if (exp.isOperation() || exp.isCall())
			OS << ")";
		else
			OS << " ";
	}

	private:
	llvm::raw_ostream& OS;
};

void ModExp::dump(llvm::raw_ostream& OS) const
{
	ModExpDumper dumper(OS);
	visit(*this, dumper);
}
