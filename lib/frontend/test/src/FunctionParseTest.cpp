#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>

using namespace std;
using namespace modelica;

TEST(FunctionTest, functionName)	// NOLINT
{
	Parser parser("function Foo end Foo;");
	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ("Foo", ast.getName());
}

TEST(FunctionTest, singleAlgorithm)	 // NOLINT
{
	Parser parser("function Foo"
								"	algorithm x := 0;"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ(1, ast.getAlgorithms().size());
}

TEST(FunctionTest, multipleAlgorithms)	// NOLINT
{
	Parser parser("function Foo"
								"	algorithm x := 0;"
								"	algorithm y := 0;"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ(2, ast.getAlgorithms().size());
}
