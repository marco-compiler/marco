#include "modelica/lowerer/Expression.hpp"

using namespace modelica;

static std::string exprKindToString(ExpressionKind kind)
{
	switch (kind)
	{
		case (ExpressionKind::zero):
			return "0";
		case (ExpressionKind::negate):
			return "-";
		case (ExpressionKind::add):
			return "+";
		case (ExpressionKind::sub):
			return "-";
		case (ExpressionKind::mult):
			return "*";
		case (ExpressionKind::divide):
			return "/";
		case (ExpressionKind::greaterThan):
			return ">";
		case (ExpressionKind::greaterEqual):
			return ">=";
		case (ExpressionKind::equal):
			return "==";
		case (ExpressionKind::different):
			return "!=";
		case (ExpressionKind::less):
			return "<";
		case (ExpressionKind::lessEqual):
			return "<=";
		case (ExpressionKind::elevation):
			return "^";
		case (ExpressionKind::module):
			return "%";
		case (ExpressionKind::conditional):
			return "?";
	}
	assert(false);	// NOLINT
	return "UNREACHABLE";
}

static void dumpOperation(const Expression& exp, llvm::raw_ostream& OS)
{
	OS << "(";
	OS << exprKindToString(exp.getKind());
	OS << " ";
}

class ExpressionDumper
{
	public:
	ExpressionDumper(llvm::raw_ostream& OS): OS(OS) {}

	void visit(const Expression& exp)
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

	void afterVisit(const Expression& exp)
	{
		if (exp.isOperation())
			OS << ")";
		else
			OS << " ";
	}

	private:
	llvm::raw_ostream& OS;
};

void Expression::dump(llvm::raw_ostream& OS) const
{
	ExpressionDumper dumper(OS);
	visit(*this, dumper);
}
