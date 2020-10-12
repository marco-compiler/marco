#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Parser.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/Type.hpp"

using namespace modelica;

TEST(ParserTest, primaryIntTest)
{
	Parser parser("4");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->getConstant().isA<int>());
}

TEST(ParserTest, primaryFloatTest)
{
	Parser parser("4.0f");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->getConstant().isA<float>());
}

TEST(ParserTest, expressionTest)
{
	Parser parser("4.0f");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->getConstant().isA<float>());
}

TEST(ParserTest, sumTest)
{
	Parser parser("4.0 + 5.0");
	auto exp = parser.arithmeticExpression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isOperation());
	EXPECT_EQ(exp->getOperation().getKind(), OperationKind::add);
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<float>());

	EXPECT_TRUE(exp->getOperation()[1].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[1].getConstant().isA<float>());
}

TEST(ParserTest, subTest)
{
	Parser parser("4.0 - 5.0");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isOperation());
	EXPECT_EQ(exp->getOperation().getKind(), OperationKind::add);
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<float>());

	EXPECT_EQ(exp->getOperation().argumentsCount(), 2);
	EXPECT_TRUE(exp->getOperation()[1].isOperation());
	EXPECT_EQ(
			exp->getOperation()[1].getOperation().getKind(), OperationKind::subtract);
}

TEST(ParserTest, andTest)
{
	Parser parser("true and false");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isOperation());
	EXPECT_EQ(exp->getOperation().getKind(), OperationKind::land);
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<bool>());

	EXPECT_TRUE(exp->getOperation()[1].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[1].getConstant().isA<bool>());
}

TEST(ParserTest, orTest)
{
	Parser parser("true or false");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isOperation());
	EXPECT_EQ(exp->getOperation().getKind(), OperationKind::lor);
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<bool>());

	EXPECT_TRUE(exp->getOperation()[1].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[1].getConstant().isA<bool>());
}

TEST(ParserTest, division)
{
	Parser parser("4.0 / 5.0");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isOperation());
	EXPECT_EQ(exp->getOperation().getKind(), OperationKind::divide);
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<float>());

	EXPECT_EQ(exp->getOperation().argumentsCount(), 2);
	EXPECT_TRUE(exp->getOperation()[1].isA<Constant>());
}

TEST(ParserTest, equation)
{
	Parser parser("4.0 / 5.0 = b * a");
	auto exp = parser.equation();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->getLeftHand().isOperation());
	EXPECT_TRUE(exp->getRightHand().isOperation());
}

TEST(ParserTest, classTest)
{
	Parser parser("model example end example;");
	auto exp = parser.classDefinition();
	if (!exp)
		FAIL();

	EXPECT_EQ(exp->getName(), "example");
}

TEST(ParserTest, memberTest)
{
	Parser parser("Real[10, 10] Qb(unit = \"W\")");
	auto exp = parser.element();
	if (!exp)
		FAIL();

	EXPECT_FALSE(exp->hasInitializer());
	EXPECT_FALSE(exp->hasStartOverload());
	EXPECT_EQ(exp->getName(), "Qb");
	EXPECT_EQ(exp->getType(), makeType<float>(10, 10));
}

TEST(ParserTest, memberStartOverloadTest)
{
	Parser parser("Real[10, 10] Qb(start = W)");
	auto exp = parser.element();
	if (!exp)
		FAIL();

	EXPECT_FALSE(exp->hasInitializer());
	EXPECT_TRUE(exp->hasStartOverload());
	EXPECT_EQ(exp->getName(), "Qb");
	EXPECT_EQ(exp->getType(), makeType<float>(10, 10));
	EXPECT_TRUE(exp->getStartOverload().isA<ReferenceAccess>());
}
