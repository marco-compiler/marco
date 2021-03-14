#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Passes.h>
#include <modelica/frontend/SymbolTable.hpp>

using namespace modelica;
using namespace std;

TEST(folderTest, sumShouldFold)
{
	Expression exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::add,
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 3),
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 4));

	ConstantFolder folder;
	if (folder.run(exp))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 7);
}

TEST(folderTest, subShouldFold)
{
	Expression exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::subtract,
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 3),
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 2));

	ConstantFolder folder;
	if (folder.run(exp))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 1);
}

TEST(folderTest, sumOfSubShouldFold)
{
	Expression exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::add,
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 3),
			Expression::constant(SourcePosition::unknown(), makeType<int>(), -1));

	ConstantFolder folder;
	if (folder.run(exp))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionShouldFold)
{
	Expression exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::add,
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 3),
			Expression::constant(SourcePosition::unknown(), makeType<int>(), -1));

	exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::subscription,
			Expression::reference(SourcePosition::unknown(), makeType<int>(10), "name"),
			exp);

	ConstantFolder folder;
	auto& symbolTable = folder.getSymbolTable();
	ConstantFolder::SymbolTableScope scope(symbolTable);

	Member m(SourcePosition::unknown(), "name", makeType<int>(10), TypePrefix::none());
	symbolTable.insert(m.getName(), Symbol(m));

	if (folder.run(exp))
		FAIL();

	EXPECT_TRUE(exp.isA<Operation>());
	auto& accessIndex = exp.get<Operation>()[1];
	EXPECT_TRUE(accessIndex.isA<Constant>());
	EXPECT_EQ(accessIndex.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionInDerShouldFold)
{
	Expression exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::add,
			Expression::constant(SourcePosition::unknown(), makeType<int>(), 3),
			Expression::constant(SourcePosition::unknown(), makeType<int>(), -1));

	exp = Expression::operation(
			SourcePosition::unknown(),
			makeType<int>(),
			OperationKind::subscription,
			Expression::reference(SourcePosition::unknown(), makeType<int>(10), "name"),
			exp);

	auto refToDer = Expression::reference(SourcePosition::unknown(), Type::unknown(), "der");
	auto call = Expression::call(SourcePosition::unknown(), Type::unknown(), move(refToDer), move(exp));

	ConstantFolder folder;
	auto& symbolTable = folder.getSymbolTable();
	ConstantFolder::SymbolTableScope scope(symbolTable);

	Member m(SourcePosition::unknown(), "name", makeType<int>(10), TypePrefix::none());
	symbolTable.insert(m.getName(), Symbol(m));

	if (folder.run(call))
		FAIL();

	EXPECT_TRUE(call.isA<Call>());
	auto& arg = call.get<Call>()[0];
	EXPECT_TRUE(arg.isA<Operation>());
	auto& accessIndex = arg.get<Operation>()[1];
	EXPECT_EQ(accessIndex.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, startDeclarationWithReference)	 // NOLINT
{
	Parser parser("model C"
								"  parameter Real A = 315.15;"
								"  Real[10, 10, 4] T(start = A);"
								"end C;");

	auto expectedAST = parser.classDefinition();

	if (!expectedAST)
		FAIL();

	auto ast = move(*expectedAST);

	ConstantFolder folder;

	if (folder.run(ast))
		FAIL();

	const auto& model = ast.get<Class>();
	bool isConstant = model.getMembers()[1]->getStartOverload().isA<Constant>();
	EXPECT_TRUE(isConstant);
}
