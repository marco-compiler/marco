#include "modelica/simulation/SimExp.hpp"

using namespace modelica;

static std::string exprKindToString(SimExpKind kind)
{
	switch (kind)
	{
		case (SimExpKind::zero):
			return "0";
		case (SimExpKind::negate):
			return "-";
		case (SimExpKind::add):
			return "+";
		case (SimExpKind::sub):
			return "-";
		case (SimExpKind::mult):
			return "*";
		case (SimExpKind::divide):
			return "/";
		case (SimExpKind::greaterThan):
			return ">";
		case (SimExpKind::greaterEqual):
			return ">=";
		case (SimExpKind::equal):
			return "==";
		case (SimExpKind::different):
			return "!=";
		case (SimExpKind::less):
			return "<";
		case (SimExpKind::lessEqual):
			return "<=";
		case (SimExpKind::elevation):
			return "^";
		case (SimExpKind::module):
			return "%";
		case (SimExpKind::conditional):
			return "?";
	}
	assert(false);	// NOLINT
	return "UNREACHABLE";
}

SimType SimExp::Operation::getOperationReturnType() const
{
	switch (kind)
	{
		case SimExpKind::zero:
			return SimType(BultinSimTypes::BOOL);
		case SimExpKind::negate:
		case SimExpKind::add:
		case SimExpKind::sub:
		case SimExpKind::mult:
		case SimExpKind::divide:
		case SimExpKind::elevation:
		case SimExpKind::module:
		case SimExpKind::conditional:
			return leftHandExpression->getSimType();
		case SimExpKind::greaterThan:
		case SimExpKind::greaterEqual:
		case SimExpKind::equal:
		case SimExpKind::different:
		case SimExpKind::less:
		case SimExpKind::lessEqual:
			return leftHandExpression->getSimType().as(BultinSimTypes::BOOL);
	}
	assert(false && "Unreachable");	 // NOLINT
	return SimType(BultinSimTypes::BOOL);
}

static void dumpOperation(const SimExp& exp, llvm::raw_ostream& OS)
{
	OS << "(";
	OS << exprKindToString(exp.getKind());
	OS << " ";
}

class SimExpDumper
{
	public:
	SimExpDumper(llvm::raw_ostream& OS): OS(OS) {}

	void visit(const SimExp& exp)
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

		dumpOperation(exp, OS);
	}

	void afterVisit(const SimExp& exp)
	{
		if (exp.isOperation())
			OS << ")";
		else
			OS << " ";
	}

	private:
	llvm::raw_ostream& OS;
};

void SimExp::dump(llvm::raw_ostream& OS) const
{
	SimExpDumper dumper(OS);
	visit(*this, dumper);
}
