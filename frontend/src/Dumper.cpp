#include "modelica/Dumper/Dumper.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

constexpr raw_ostream::Colors mainColor = raw_ostream::Colors::CYAN;
constexpr raw_ostream::Colors errorColor = raw_ostream::Colors::RED;
constexpr raw_ostream::Colors secondaryColor = raw_ostream::Colors::GREEN;

string toString(ComponentClause::FlowStream flow)
{
	switch (flow)
	{
		case ComponentClause::FlowStream::flow:
			return "flow";
		case ComponentClause::FlowStream::stream:
			return "stream";
		case ComponentClause::FlowStream::none:
			return "none";
	}

	return "Unreachable";
}

string toString(ComponentClause::IO io)
{
	switch (io)
	{
		case ComponentClause::IO::input:
			return "input";
		case ComponentClause::IO::output:
			return "output";
		case ComponentClause::IO::none:
			return "none";
	}

	return "Unreachable";
}

string toString(ComponentClause::Type tp)
{
	switch (tp)
	{
		case ComponentClause::Type::discrete:
			return "discrete";
		case ComponentClause::Type::parameter:
			return "parameter";
		case ComponentClause::Type::consant:
			return "constant";
		case ComponentClause::Type::none:
			return "none";
	}

	return "Unreachable";
}

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

string toString(ClassDecl::SubType subType)
{
	switch (subType)
	{
		case (ClassDecl::SubType::Class):
			return "Class";
		case (ClassDecl::SubType::Model):
			return "Model";
		case (ClassDecl::SubType::Record):
			return "Record";
		case (ClassDecl::SubType::OperatorRecord):
			return "Operator Record";
		case (ClassDecl::SubType::Block):
			return "Block";
		case (ClassDecl::SubType::Operator):
			return "Operator";
		case (ClassDecl::SubType::Connector):
			return "Connector";
		case (ClassDecl::SubType::ExpandableConnector):
			return "ExpandableConnector";
		case (ClassDecl::SubType::Type):
			return "Type";
		case (ClassDecl::SubType::Package):
			return "Package";
		case (ClassDecl::SubType::Function):
			return "Function";
		case (ClassDecl::SubType::OperatorFunction):
			return "Operator Function";
	};
	return "UNKOWN SUBTYPE";
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

	auto visit(unique_ptr<Declaration> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Decl comment" << decl->getComment() << "\n";

		return decl;
	}

	auto visit(unique_ptr<CompositeDecl> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Composite Decl\n";

		indentations++;
		return decl;
	}

	auto visit(unique_ptr<ExprCompositeDecl> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Expr Composite Decl\n";

		indentations++;
		return decl;
	}

	auto visit(unique_ptr<EqCompositeDecl> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Eq Composite Decl\n";

		indentations++;
		return decl;
	}

	auto visit(unique_ptr<StatementCompositeDecl> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Statement Composite Decl\n";

		indentations++;
		return decl;
	}

	auto visit(unique_ptr<ClassDecl> decl)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Class Decl ";
		OS.changeColor(secondaryColor);
		OS << decl->getName() << " ";
		if (decl->isPure())
			OS << "pure ";
		if (decl->isPartial())
			OS << "partial ";
		if (decl->isEncapsulated())
			OS << "encapsulated";
		OS.changeColor(mainColor);
		OS << "Subtype: ";
		OS.changeColor(secondaryColor);
		OS << toString(decl->subType());
		OS << "\n";

		indentations++;
		return decl;
	}

	auto visit(unique_ptr<SimpleModification> modification)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Simple Modification\n";

		indentations++;
		return modification;
	}

	auto visit(unique_ptr<ClassModification> modification)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Class Modification\n";

		indentations++;
		return modification;
	}

	auto visit(unique_ptr<Annotation> modification)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Annotation\n";

		indentations++;
		return modification;
	}

	auto visit(unique_ptr<OverridingClassModification> modification)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Overriding class modification \n";

		indentations++;
		return modification;
	}

	auto visit(unique_ptr<CompositionSection> modification)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Composition Section \n";

		indentations++;
		return modification;
	}

	auto visit(unique_ptr<ComponentDeclaration> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Component Declaration";
		OS.changeColor(secondaryColor);
		OS << "Name " << declaration->getIdent() << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ComponentClause> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Component clause ";
		OS.changeColor(secondaryColor);
		OS << "Name ";
		if (declaration->hasGlobalLookup())
			OS << ".";

		for (const auto& nm : declaration->getName())
			OS << nm << ".";

		OS << " ";

		OS << "Flowstream: " << toString(declaration->getPrefix().getFlowStream());

		OS << " IO: " << toString(declaration->getPrefix().getIOType());
		OS << " Type: " << toString(declaration->getPrefix().getType());

		OS << "\n";
		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ElementModification> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Element modification ";
		OS.changeColor(secondaryColor);
		OS << "Name ";

		for (const auto& nm : declaration->getName())
			OS << nm << ".";

		OS << " ";

		if (declaration->isFinal())
			OS << "final ";

		if (declaration->hasEach())
			OS << "each ";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ReplecableModification> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Replaceable modification ";
		OS.changeColor(secondaryColor);

		if (declaration->isFinal())
			OS << "final ";

		if (declaration->hasEach())
			OS << "each ";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<Element> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Element ";
		OS.changeColor(secondaryColor);

		if (declaration->isInner())
			OS << "Inner ";

		if (declaration->isOuter())
			OS << "Outer ";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<Redeclaration> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Element ";
		OS.changeColor(secondaryColor);

		if (declaration->isFinal())
			OS << "final ";

		if (declaration->hasEach())
			OS << "each ";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ConstrainingClause> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Constraining Clause ";
		OS.changeColor(secondaryColor);

		OS << "Name: ";

		const TypeSpecifier& spec = declaration->getTypeSpecifier();

		if (spec.second)
			OS << ".";

		for (const auto& nm : spec.first)
			OS << nm << ".";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<EnumerationLiteral> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Enumeration Literal ";
		OS.changeColor(secondaryColor);

		OS << "Name: ";
		OS << declaration->getName();
		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ExtendClause> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Extend Clause ";
		OS.changeColor(secondaryColor);

		const TypeSpecifier& spec = declaration->getTypeSpecifier();

		if (spec.second)
			OS << ".";

		for (const auto& nm : spec.first)
			OS << nm << ".";

		OS << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<Composition> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Composition\n";
		OS.changeColor(secondaryColor);

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ArraySubscriptionDecl> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Array Subscription Declaration\n";
		OS.changeColor(secondaryColor);

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ExternalFunctionCall> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "External Function Call";
		OS.changeColor(secondaryColor);
		OS << "name: " << declaration->getName() << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ConditionAttribute> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Condition Attribute\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<EquationSection> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Equation Section";
		OS.changeColor(secondaryColor);
		OS << "Initial: " << std::to_string(declaration->isInitial()) << "\n";

		indentations++;
		return declaration;
	}
	auto visit(unique_ptr<AlgorithmSection> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Alghortim Section";
		OS.changeColor(secondaryColor);
		OS << "Initial: " << std::to_string(declaration->isInitial()) << "\n";

		indentations++;
		return declaration;
	}
	auto visit(unique_ptr<DerClassDecl> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Der Class Declaration";
		OS.changeColor(secondaryColor);

		const TypeSpecifier& spec = declaration->getTypeSpecifier();

		if (spec.second)
			OS << ".";

		for (const auto& nm : spec.first)
			OS << nm << ".";

		OS << " Args: ";

		const auto& idents = declaration->getIdents();

		for (const auto& nm : idents)
			OS << nm << ".";

		unique_ptr<ClassDecl> decl = move(declaration);
		decl = visit(move(decl));

		return llvm::cast<DerClassDecl>(move(decl));
	}

	auto visit(unique_ptr<LongClassDecl> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Long Class Declaration\n";
		OS.changeColor(secondaryColor);

		unique_ptr<ClassDecl> decl = move(declaration);
		decl = visit(move(decl));

		return llvm::cast<LongClassDecl>(move(decl));
	}

	auto visit(unique_ptr<ImportClause> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Import Clause ";
		OS.changeColor(secondaryColor);
		OS << "import all: " << std::to_string(declaration->importAllNamespace())
			 << "\n";

		indent();
		if (!declaration->getNewName().empty())
			OS << "new name " << declaration->getNewName();

		OS << "import all: " << std::to_string(declaration->importAllNamespace());

		OS << "base name: ";

		for (const auto& nm : declaration->getBaseName())
			OS << nm << ".";

		if (!declaration->namesToImport().empty())
		{
			OS << "names to import: ";
			for (const auto& name : declaration->namesToImport())
				OS << name << " ";
		}

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ElementList> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Elment List\n";
		OS.changeColor(secondaryColor);
		OS << "protected: " << std::to_string(declaration->isProtected()) << "\n";

		indentations++;
		return declaration;
	}

	auto visit(unique_ptr<ShortClassDecl> declaration)
	{
		indent();

		OS.changeColor(mainColor);
		OS << "Short Class Declaration";
		OS.changeColor(secondaryColor);
		if (declaration->isInput())
			OS << "Input ";

		if (declaration->isOutput())
			OS << "Output ";

		const TypeSpecifier& spec = declaration->getTypeSpecifier();

		if (spec.second)
			OS << ".";

		for (const auto& nm : spec.first)
			OS << nm << ".";

		OS << "\n";

		unique_ptr<ClassDecl> decl = move(declaration);
		decl = visit(move(decl));

		return llvm::cast<ShortClassDecl>(move(decl));
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

template<typename Node>
auto dumpImpl(unique_ptr<Node> node, raw_ostream& OS)
{
	DumperVisitor visitor(OS);
	return topDownVisit(move(node), visitor);
}

unique_ptr<Statement> modelica::dump(
		std::unique_ptr<Statement> stmt, llvm::raw_ostream& OS)
{
	return dumpImpl(move(stmt), OS);
}
unique_ptr<Declaration> modelica::dump(
		std::unique_ptr<Declaration> decl, llvm::raw_ostream& OS)
{
	return dumpImpl(move(decl), OS);
}
unique_ptr<Expr> modelica::dump(
		std::unique_ptr<Expr> exp, llvm::raw_ostream& OS)
{
	return dumpImpl(move(exp), OS);
}
unique_ptr<Equation> modelica::dump(
		std::unique_ptr<Equation> eq, llvm::raw_ostream& OS)
{
	return dumpImpl(move(eq), OS);
}
