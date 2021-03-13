#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;

TEST(Parser, emptyAlgorithm)	 // NOLINT
{
	Parser parser("algorithm");

	auto ast = parser.algorithmSection();
	ASSERT_FALSE(!ast);

	EXPECT_TRUE(ast->getStatements().empty());
}

TEST(Parser, algorithmStatementsCount)	// NOLINT
{
	Parser parser("algorithm"
								"	x := 1;"
								"	y := 2;"
								"	z := 3;");

	auto ast = parser.algorithmSection();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->getStatements().size(), 3);
}
