#include "gtest/gtest.h"

#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/ConstantFolder.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/Parser.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"

using namespace std;
using namespace modelica;

TEST(folderTest, sumShouldFold)
{
	Expression exp = Expression::op<OperationKind::add>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), 4));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.getConstant().get<BuiltinType::Integer>(), 7);
}

TEST(folderTest, subShouldFold)
{
	Expression exp = Expression::op<OperationKind::subtract>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), 2));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.getConstant().get<BuiltinType::Integer>(), 1);
}

TEST(folderTest, sumOfSubShouldFold)
{
	Expression exp = Expression::op<OperationKind::add>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), -1));
	ConstantFolder folder;
	if (folder.fold(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.getConstant().get<BuiltinType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionShouldFold)
{
	Expression exp = Expression::op<OperationKind::add>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), -1));
	exp = Expression::op<OperationKind::subscription>(
			makeType<int>(),
			Expression(makeType<int>(10), ReferenceAccess("name")),
			exp);
	ConstantFolder folder;

	SymbolTable t;
	Member m("name", makeType<int>(10), false);
	t.addSymbol(m);
	if (folder.fold(exp, t))
		FAIL();
	Call

			EXPECT_TRUE(exp.isOperation());
	auto& accessIndex = exp.getOperation()[1];
	EXPECT_TRUE(accessIndex.isA<Constant>());
	EXPECT_EQ(accessIndex.getConstant().get<BuiltinType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionInDerShouldFold)
{
	Expression exp = Expression::op<OperationKind::add>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), -1));
	exp = Expression::op<OperationKind::subscription>(
			makeType<int>(),
			Expression(makeType<int>(10), ReferenceAccess("name")),
			exp);

	auto refToDer = Expression(Type::unknown(), ReferenceAccess("der"));
	auto call = makeCall(move(refToDer), { move(exp) });
	ConstantFolder folder;

	SymbolTable t;
	Member m("name", makeType<int>(10), false);
	t.addSymbol(m);
	if (folder.fold(call, t))
		FAIL();

	EXPECT_TRUE(call.isA<Call>());
	auto& arg = call.get<Call>()[0];
	EXPECT_TRUE(arg.isOperation());
	auto& accessIndex = arg.getOperation()[1];
	EXPECT_EQ(accessIndex.getConstant().get<BuiltinType::Integer>(), 2);
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

	bool isConstant = ast.getMembers()[1].getStartOverload().isA<Constant>();
	EXPECT_TRUE(isConstant);
}
