#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace std;
using namespace modelica;

TEST(StatementTest, integerAssignment)	// NOLINT
{
	Parser parser("x := 1;");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	auto destinations = ast.getDestinations();
	EXPECT_EQ(1, destinations.size());
	EXPECT_TRUE(destinations[0]->isA<ReferenceAccess>());

	// Right-hand side
	auto& expression = ast.getExpression();
	EXPECT_TRUE(expression.get<Constant>().isA<BuiltInType::Integer>());
}

TEST(StatementTest, floatAssignment)	// NOLINT
{
	Parser parser("x := 3.14;");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	auto destinations = ast.getDestinations();
	EXPECT_EQ(1, destinations.size());
	EXPECT_TRUE(destinations[0]->isA<ReferenceAccess>());

	// Right-hand side
	auto& expression = ast.getExpression();
	EXPECT_TRUE(expression.get<Constant>().isA<BuiltInType::Float>());
}

TEST(StatementTest, referenceAssignment)	// NOLINT
{
	Parser parser("x := y;");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side
	EXPECT_EQ(1, ast.getDestinations().size());
	auto destinations = ast.getDestinations()[0];
	EXPECT_EQ("x", destinations->get<ReferenceAccess>().getName());

	// Right-hand side
	auto& expression = ast.getExpression();
	EXPECT_EQ("y", expression.get<ReferenceAccess>().getName());
}

TEST(StatementTest, functionCall)	 // NOLINT
{
	Parser parser("x := Foo (y, z);");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Left-hand side is not tested because not so important for this test
	auto& call = ast.getExpression().get<Call>();

	// Function name
	EXPECT_TRUE(call.getFunction().isA<ReferenceAccess>());

	// Function parameters
	EXPECT_EQ(2, call.argumentsCount());
	EXPECT_EQ("y", call[0].get<ReferenceAccess>().getName());
	EXPECT_EQ("z", call[1].get<ReferenceAccess>().getName());
}

TEST(StatementTest, multipleOutputs)	// NOLINT
{
	Parser parser("(x, y) := Foo ();");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Right-hand side is not tested because not so important for this test
	auto destinations = ast.getDestinations();
	EXPECT_EQ(2, destinations.size());
	EXPECT_EQ("x", destinations[0]->get<ReferenceAccess>().getName());
	EXPECT_EQ("y", destinations[1]->get<ReferenceAccess>().getName());
}

TEST(StatementTest, ignoredOutputs)	 // NOLINT
{
	Parser parser("(x, , z) := Foo ();");

	auto expectedAst = parser.assignmentStatement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	// Right-hand side is not tested because not so important for this test
	auto destinations = ast.getDestinations();
	EXPECT_EQ(3, destinations.size());

	EXPECT_EQ("x", destinations[0]->get<ReferenceAccess>().getName());
	EXPECT_TRUE(destinations[1]->get<ReferenceAccess>().isDummy());
	EXPECT_EQ("z", destinations[2]->get<ReferenceAccess>().getName());
}
