#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Parser.hpp"

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

	EXPECT_TRUE(exp->isOperation());
	EXPECT_TRUE(exp->getOperation()[0].isA<Constant>());
	EXPECT_TRUE(exp->getOperation()[0].getConstant().isA<float>());
}
