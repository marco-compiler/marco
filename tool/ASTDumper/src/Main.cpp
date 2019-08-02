#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/AST/Visitor.hpp"
#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

cl::opt<string> InputFileName(
		cl::Positional, cl::desc("<input-file>"), cl::init("-"));

ExitOnError exitOnErr;
constexpr raw_ostream::Colors mainColor = raw_ostream::Colors::CYAN;
constexpr raw_ostream::Colors errorColor = raw_ostream::Colors::RED;
constexpr raw_ostream::Colors secondaryColor = raw_ostream::Colors::GREEN;

string toString(BinaryExprOp op)
{
	switch (op)
	{
		case (BinaryExprOp::Sum):
			return "+";
		case (BinaryExprOp::Subtraction):
			return "-";
		case (BinaryExprOp::Multiply):
			return "*";
		case (BinaryExprOp::Division):
			return "/";
		case (BinaryExprOp::LogicalOr):
			return "OR";
		case (BinaryExprOp::LogicalAnd):
			return "AND";
		case (BinaryExprOp::LessEqual):
			return "<=";
		case (BinaryExprOp::Less):
			return "<";
		case (BinaryExprOp::Equal):
			return "==";
		case (BinaryExprOp::Greater):
			return ">";
		case (BinaryExprOp::GreatureEqual):
			return ">=";
		case (BinaryExprOp::Different):
			return "!=";
		case (BinaryExprOp::PowerOf):
			return "^";
	}
	return "UNKOWN OP";
}

string toString(UnaryExprOp op)
{
	switch (op)
	{
		case (UnaryExprOp::LogicalNot):
			return "NOT";
		case (UnaryExprOp::Plus):
			return "+";
		case (UnaryExprOp::Minus):
			return "-";
	};
	return "UNKOWN OP";
}

class DumperVisitor
{
	public:
	DumperVisitor(raw_ostream& os): OS(os), indentations(0) {}
	template<typename T>
	std::unique_ptr<T> visit(std::unique_ptr<T> ptr)
	{
		indentations++;
		llvm::errs().changeColor(errorColor);
		llvm::errs() << "Unsupperted AST node in " << __PRETTY_FUNCTION__ << "\n";
		return ptr;
	}

	template<typename T>
	void afterChildrenVisit(T*)
	{
		indentations--;
	}

	auto visit(unique_ptr<IntLiteralExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "int: ";
		OS.changeColor(secondaryColor);
		OS << expr->getValue() << "\n";
		return expr;
	}

	auto visit(unique_ptr<BoolLiteralExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "bool : ";
		OS.changeColor(secondaryColor);
		OS << expr->getValue() << "\n";
		return expr;
	}

	auto visit(unique_ptr<StringLiteralExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "String: ";
		OS.changeColor(secondaryColor);
		OS << expr->getValue() << "\n";
		return expr;
	}

	auto visit(unique_ptr<IfEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "If Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<IfStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "If Statement:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<CompositeEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Composite Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<CompositeStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Composite Statement:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<WhenEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "When Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<WhenStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "When Statement:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<WhileStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "While Statement:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ForEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "For Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ForStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "For Statement:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<CallEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Call Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<CallStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Call Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<AssignStatement> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Assign Equation:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ConnectClause> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Connect Clause:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<FloatLiteralExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Float: ";
		OS.changeColor(secondaryColor);
		OS << expr->getValue() << "\n";
		return expr;
	}

	auto visit(unique_ptr<SimpleEquation> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Simple Equation: \n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ExprList> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Expr List: \n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<EndExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "End\n";
		return expr;
	}

	auto visit(unique_ptr<BinaryExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Binary expr ";
		OS.changeColor(secondaryColor);
		OS << toString(expr->getOpCode()) << "\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<UnaryExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Unary expr ";
		OS.changeColor(secondaryColor);
		OS << toString(expr->getOpCode()) << "\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<RangeExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Range expr:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<DirectArrayConstructorExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Array:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ArraySubscriptionExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Subscript Expr:\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<AcceptAllExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Accept All Expr:\n";
		return expr;
	}

	auto visit(unique_ptr<NamedArgumentExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Named Argument ";
		OS.changeColor(secondaryColor);
		OS << expr->getName() << ":\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<IfElseExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "if else Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ArrayConcatExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Array Concat Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<DerFunctionCallExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Der Call Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<InitialFunctionCallExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Intial Call Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<PureFunctionCallExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Pure Call Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ComponentFunctionCallExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Component Call Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<PartialFunctioCallExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Partial Call Expr\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ComponentReferenceExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		if (expr->hasGlobalLookup())
			OS << "Global Component ";
		else
			OS << "Component ";
		OS.changeColor(secondaryColor);
		OS << expr->getName() << "\n";
		indentations++;
		return expr;
	}

	auto visit(unique_ptr<ForInArrayConstructorExpr> expr)
	{
		indent();
		OS.changeColor(mainColor);
		OS << "Range expr:\n";

		for (unsigned a = 0; a < expr->forInCount(); a++)
		{
			indent();
			OS.changeColor(mainColor);
			OS << "Name ";
			OS.changeColor(secondaryColor);
			OS << expr->getDeclaredName(a) << "\n";
		}

		indentations++;
		return expr;
	}

	void indent()
	{
		for (int a = 0; a < indentations; a++)
			OS << " ";
	}

	private:
	raw_ostream& OS;
	int indentations;
};

int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	auto ast = exitOnErr(parser.equation());
	DumperVisitor visitor(outs());
	ast = topDownVisit(move(ast), visitor);

	return 0;
}
