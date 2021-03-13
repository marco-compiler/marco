#include <gtest/gtest.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Statement.hpp>

using namespace modelica;

TEST(AST, singleElementTupleCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression exp = Expression::reference(location, makeType<int>(), "x");
	Tuple tuple(location, { exp });
	EXPECT_EQ(tuple.size(), 1);
}

TEST(AST, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression exp1 = Expression::reference(location, makeType<int>(), "x");
	Expression exp2 = Expression::reference(location, makeType<float>(), "y");
	Expression exp3 = Expression::reference(location, Type::unknown(), "z");

	Tuple tuple(location, { exp1, exp2, exp3 });

	EXPECT_EQ(tuple.size(), 3);
}

TEST(Parser, emptyTuple)	 // NOLINT
{
	Parser parser("() := Foo(time);");

	auto ast = parser.statement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->get<AssignmentStatement>().getDestinations().size(), 0);
}
