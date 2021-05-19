#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace frontend;

TEST(ParserTest, primaryIntTest)
{
	Parser parser("4");
	auto expression = parser.primary();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Constant>()->isa<BuiltInType::Integer>());
}

TEST(ParserTest, primaryFloatTest)
{
	Parser parser("4.0f");
	auto expression = parser.primary();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Constant>()->isa<BuiltInType::Float>());
}

TEST(ParserTest, expressionTest)
{
	Parser parser("4.0f");
	auto expression = parser.expression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Constant>()->isa<BuiltInType::Float>());
}

TEST(ParserTest, sumTest)
{
	Parser parser("4.0 + 5.0");
	auto expression = parser.arithmeticExpression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getOperationKind(), OperationKind::add);

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->get<Constant>()->isa<BuiltInType::Float>());

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->get<Constant>()->isa<BuiltInType::Float>());
}

TEST(ParserTest, subTest)
{
	Parser parser("4.0 - 5.0");
	auto expression = parser.expression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getOperationKind(), OperationKind::add);

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->get<Constant>()->isa<BuiltInType::Float>());

	EXPECT_EQ((*expression)->get<Operation>()->argumentsCount(), 2);
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getArg(1)->get<Operation>()->getOperationKind(), OperationKind::subtract);
}

TEST(ParserTest, andTest)
{
	Parser parser("true and false");
	auto expression = parser.expression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getOperationKind(), OperationKind::land);

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->get<Constant>()->isa<BuiltInType::Boolean>());

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->get<Constant>()->isa<BuiltInType::Boolean>());
}

TEST(ParserTest, orTest)
{
	Parser parser("true or false");
	auto expression = parser.expression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getOperationKind(), OperationKind::lor);

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->get<Constant>()->isa<BuiltInType::Boolean>());

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->get<Constant>()->isa<BuiltInType::Boolean>());
}

TEST(ParserTest, division)
{
	Parser parser("4.0 / 5.0");
	auto expression = parser.expression();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->isa<Operation>());
	EXPECT_EQ((*expression)->get<Operation>()->getOperationKind(), OperationKind::divide);

	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->isa<Constant>());
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(0)->get<Constant>()->isa<BuiltInType::Float>());

	EXPECT_EQ((*expression)->get<Operation>()->argumentsCount(), 2);
	EXPECT_TRUE((*expression)->get<Operation>()->getArg(1)->isa<Constant>());
}

TEST(ParserTest, equation)
{
	Parser parser("4.0 / 5.0 = b * a");
	auto expression = parser.equation();

	if (!expression || !*expression)
		FAIL();

	EXPECT_TRUE((*expression)->getLhsExpression()->isa<Operation>());
	EXPECT_TRUE((*expression)->getRhsExpression()->isa<Operation>());
}

TEST(ParserTest, classTest)
{
	Parser parser("model example end example;");
	auto cls = parser.classDefinition();

	if (!cls || !*cls)
		FAIL();

	EXPECT_EQ((*cls)->get<Model>()->getName(), "example");
}

TEST(ParserTest, memberTest)
{
	Parser parser("Real[10, 10] Qb(unit = \"W\")");
	auto member = parser.element();

	if (!member || !*member)
		FAIL();

	EXPECT_FALSE((*member)->hasInitializer());
	EXPECT_FALSE((*member)->hasStartOverload());
	EXPECT_EQ((*member)->getName(), "Qb");
	EXPECT_EQ((*member)->getType(), makeType<BuiltInType::Float>(10, 10));
}

TEST(ParserTest, memberStartOverloadTest)
{
	Parser parser("Real[10, 10] Qb(start = W)");
	auto member = parser.element();

	if (!member || !*member)
		FAIL();

	EXPECT_FALSE((*member)->hasInitializer());
	EXPECT_TRUE((*member)->hasStartOverload());
	EXPECT_EQ((*member)->getName(), "Qb");
	EXPECT_EQ((*member)->getType(), makeType<BuiltInType::Float>(10, 10));
	EXPECT_TRUE((*member)->getStartOverload()->isa<ReferenceAccess>());
}
