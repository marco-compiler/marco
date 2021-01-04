#include <gtest/gtest.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Tuple.hpp>
#include <vector>

using namespace modelica;
using namespace std;

TEST(TupleTest, emptyTuple)	 // NOLINT
{
	Parser parser("() := Foo(time);");

	auto expectedAst = parser.statement();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
}

TEST(TupleTest, singleElementTupleCanBeBuilt)	 // NOLINT
{
	Expression exp = Expression::reference(SourcePosition::unknown(), Type::Int(), "x");
	Tuple tuple(SourcePosition::unknown(), { exp });
	EXPECT_EQ(1, tuple.size());
}

TEST(TupleTest, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	Expression exp1 = Expression::reference(SourcePosition::unknown(), Type::Int(), "x");
	Expression exp2 = Expression::reference(SourcePosition::unknown(), Type::Float(), "y");
	Expression exp3 = Expression::reference(SourcePosition::unknown(), Type::unknown(), "z");
	Tuple tuple(SourcePosition::unknown(), { exp1, exp2, exp3 });
	EXPECT_EQ(3, tuple.size());
}
