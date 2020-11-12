#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace std;

TEST(AlgorithmTest, emptyAlgorithm)	 // NOLINT
{
	Parser parser("algorithm");
	auto expectedAst = parser.algorithmSection();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_TRUE(ast.getStatements().empty());
}

TEST(AlgorithmTest, statementsCount)	// NOLINT
{
	Parser parser("algorithm"
								"	x := 1;"
								"	y := 2;"
								"	z := 3;");

	auto expectedAst = parser.algorithmSection();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ(3, ast.getStatements().size());
}
