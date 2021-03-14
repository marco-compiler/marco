#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;

TEST(ParserTest, primaryIntTest)
{
	Parser parser("4");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->get<Constant>().isA<BuiltInType::Integer>());
}

TEST(ParserTest, primaryFloatTest)
{
	Parser parser("4.0f");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->get<Constant>().isA<BuiltInType::Float>());
}

TEST(ParserTest, expressionTest)
{
	Parser parser("4.0f");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Constant>());
	EXPECT_TRUE(exp->get<Constant>().isA<BuiltInType::Float>());
}

TEST(ParserTest, sumTest)
{
	Parser parser("4.0 + 5.0");
	auto exp = parser.arithmeticExpression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Operation>());
	EXPECT_EQ(exp->get<Operation>().getKind(), OperationKind::add);
	EXPECT_TRUE(exp->get<Operation>()[0].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[0].get<Constant>().isA<BuiltInType::Float>());

	EXPECT_TRUE(exp->get<Operation>()[1].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[1].get<Constant>().isA<BuiltInType::Float>());
}

TEST(ParserTest, subTest)
{
	Parser parser("4.0 - 5.0");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Operation>());
	EXPECT_EQ(exp->get<Operation>().getKind(), OperationKind::add);
	EXPECT_TRUE(exp->get<Operation>()[0].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[0].get<Constant>().isA<BuiltInType::Float>());

	EXPECT_EQ(exp->get<Operation>().argumentsCount(), 2);
	EXPECT_TRUE(exp->get<Operation>()[1].isA<Operation>());
	EXPECT_EQ(
			exp->get<Operation>()[1].get<Operation>().getKind(),
			OperationKind::subtract);
}

TEST(ParserTest, andTest)
{
	Parser parser("true and false");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Operation>());
	EXPECT_EQ(exp->get<Operation>().getKind(), OperationKind::land);
	EXPECT_TRUE(exp->get<Operation>()[0].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[0].get<Constant>().isA<BuiltInType::Boolean>());

	EXPECT_TRUE(exp->get<Operation>()[1].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[1].get<Constant>().isA<BuiltInType::Boolean>());
}

TEST(ParserTest, orTest)
{
	Parser parser("true or false");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Operation>());
	EXPECT_EQ(exp->get<Operation>().getKind(), OperationKind::lor);
	EXPECT_TRUE(exp->get<Operation>()[0].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[0].get<Constant>().isA<BuiltInType::Boolean>());

	EXPECT_TRUE(exp->get<Operation>()[1].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[1].get<Constant>().isA<BuiltInType::Boolean>());
}

TEST(ParserTest, division)
{
	Parser parser("4.0 / 5.0");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->isA<Operation>());
	EXPECT_EQ(exp->get<Operation>().getKind(), OperationKind::divide);
	EXPECT_TRUE(exp->get<Operation>()[0].isA<Constant>());
	EXPECT_TRUE(
			exp->get<Operation>()[0].get<Constant>().isA<BuiltInType::Float>());

	EXPECT_EQ(exp->get<Operation>().argumentsCount(), 2);
	EXPECT_TRUE(exp->get<Operation>()[1].isA<Constant>());
}

TEST(ParserTest, equation)
{
	Parser parser("4.0 / 5.0 = b * a");
	auto exp = parser.equation();
	if (!exp)
		FAIL();

	EXPECT_TRUE(exp->getLeftHand().isA<Operation>());
	EXPECT_TRUE(exp->getRightHand().isA<Operation>());
}

TEST(ParserTest, classTest)
{
	Parser parser("model example end example;");
	auto exp = parser.classDefinition();
	if (!exp)
		FAIL();

	const auto& model = exp->get<Class>();
	EXPECT_EQ(model.getName(), "example");
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
	EXPECT_EQ(exp->getType(), makeType<BuiltInType::Float>(10, 10));
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
	EXPECT_EQ(exp->getType(), makeType<BuiltInType::Float>(10, 10));
	EXPECT_TRUE(exp->getStartOverload().isA<ReferenceAccess>());
}
