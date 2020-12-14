#include "gtest/gtest.h"

#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/ConstantFolder.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/Parser.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"

using namespace modelica;
using namespace std;

TEST(folderTest, sumShouldFold)
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location,
			makeType<int>(),
			Expression(location, makeType<int>(), 3),
			Expression(location, makeType<int>(), 4));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 7);
}

TEST(folderTest, subShouldFold)
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::subtract>(
			location,
			makeType<int>(),
			Expression(location, makeType<int>(), 3),
			Expression(location, makeType<int>(), 2));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 1);
}

TEST(folderTest, sumOfSubShouldFold)
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location,
			makeType<int>(),
			Expression(location, makeType<int>(), 3),
			Expression(location, makeType<int>(), -1));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionShouldFold)
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location, makeType<int>(),
			Expression(location, makeType<int>(), 3),
			Expression(location, makeType<int>(), -1));
	exp = Expression::op<OperationKind::subscription>(
			location, makeType<int>(),
			Expression(location, makeType<int>(10), ReferenceAccess("name")),
			exp);
	ConstantFolder folder;

	SymbolTable t;
	Member m("name", makeType<int>(10), TypePrefix::none());
	t.addSymbol(m);
	if (folder.fold(exp, t))
		FAIL();

	EXPECT_TRUE(exp.isA<Operation>());
	auto& accessIndex = exp.get<Operation>()[1];
	EXPECT_TRUE(accessIndex.isA<Constant>());
	EXPECT_EQ(accessIndex.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionInDerShouldFold)
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location, makeType<int>(),
			Expression(location, makeType<int>(), 3),
			Expression(location, makeType<int>(), -1));
	exp = Expression::op<OperationKind::subscription>(
			location, makeType<int>(),
			Expression(location, makeType<int>(10), ReferenceAccess("name")),
			exp);

	auto refToDer = Expression(location, Type::unknown(), ReferenceAccess("der"));
	auto call = makeCall(location, move(refToDer), { move(exp) });
	ConstantFolder folder;

	SymbolTable t;
	Member m("name", makeType<int>(10), TypePrefix::none());
	t.addSymbol(m);
	if (folder.fold(call, t))
		FAIL();

	EXPECT_TRUE(call.isA<Call>());
	auto& arg = call.get<Call>()[0];
	EXPECT_TRUE(arg.isA<Operation>());
	auto& accessIndex = arg.get<Operation>()[1];
	EXPECT_EQ(accessIndex.get<Constant>().get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, startDeclarationWithReference)	 // NOLINT
{
	Parser parser("model C parameter Real A = 315.15; Real[10, 10, 4] T(start = "
								"A); end C;");

	auto expectedAST = parser.classDefinition();
	if (!expectedAST)
		FAIL();

	auto ast = move(*expectedAST);

	ConstantFolder folder;
	if (folder.fold(ast, {}))
		FAIL();

	const auto& model = ast.get<Class>();
	bool isConstant = model.getMembers()[1].getStartOverload().isA<Constant>();
	EXPECT_TRUE(isConstant);
}
