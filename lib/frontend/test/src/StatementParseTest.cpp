#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace std;
using namespace modelica;

TEST(StatementTest, integerAssignment)	// NOLINT
{
	Parser parser("x := 1;");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	ASSERT_EQ(1, ast.getDestinations().size());
	ASSERT_TRUE(ast.getDestinations()[0].isA<ReferenceAccess>());

	// Right-hand side
	ASSERT_TRUE(ast.getExpression().isA<Constant>());
	ASSERT_TRUE(ast.getExpression().getConstant().isA<BuiltinType::Integer>());
}

TEST(StatementTest, floatAssignment)	// NOLINT
{
	Parser parser("x := 3.14;");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	ASSERT_EQ(1, ast.getDestinations().size());
	ASSERT_TRUE(ast.getDestinations()[0].isA<ReferenceAccess>());

	// Right-hand side
	ASSERT_TRUE(ast.getExpression().isA<Constant>());
	ASSERT_TRUE(ast.getExpression().getConstant().isA<BuiltinType::Float>());
}

TEST(StatementTest, referenceAssignment)	// NOLINT
{
	Parser parser("x := y;");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	ASSERT_EQ(1, ast.getDestinations().size());
	auto& left = ast.getDestinations()[0];
	ASSERT_TRUE(left.isA<ReferenceAccess>());
	ASSERT_EQ("x", left.get<ReferenceAccess>().getName());

	// Right-hand side
	auto& right = ast.getExpression();
	ASSERT_TRUE(right.isA<ReferenceAccess>());
	ASSERT_EQ("y", right.get<ReferenceAccess>().getName());
}

TEST(StatementTest, functionCall)	 // NOLINT
{
	Parser parser("x := Foo (y, z);");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side is not tested because not so important for this test
	auto& right = ast.getExpression();
	ASSERT_TRUE(right.isA<Call>());
	auto& call = right.get<Call>();

	// Function name
	ASSERT_TRUE(call.getFunction().isA<ReferenceAccess>());

	// Function parameters
	ASSERT_EQ(2, call.argumentsCount());
	ASSERT_TRUE(call[0].isA<ReferenceAccess>());
	ASSERT_EQ("y", call[0].get<ReferenceAccess>().getName());
	ASSERT_TRUE(call[1].isA<ReferenceAccess>());
	ASSERT_EQ("z", call[1].get<ReferenceAccess>().getName());
}

TEST(StatementTest, multipleOutputs)	// NOLINT
{
	Parser parser("(x, y) := Foo ();");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Right-hand side is not tested because not so important for this test
	auto& left = ast.getDestinations();
	ASSERT_EQ(2, left.size());

	ASSERT_TRUE(left[0].isA<ReferenceAccess>());
	ASSERT_EQ("x", left[0].get<ReferenceAccess>().getName());

	ASSERT_TRUE(left[1].isA<ReferenceAccess>());
	ASSERT_EQ("y", left[1].get<ReferenceAccess>().getName());
}
