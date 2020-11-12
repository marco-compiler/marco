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
	Expression exp(Type::Int(), ReferenceAccess("x"));
	Tuple tuple({ exp });
	EXPECT_EQ(1, tuple.size());
}

TEST(TupleTest, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	Expression exp1(Type::Int(), ReferenceAccess("x"));
	Expression exp2(Type::Float(), ReferenceAccess("y"));
	Expression exp3(Type::unknown(), ReferenceAccess("z"));
	Tuple tuple({ exp1, exp2, exp3 });
	EXPECT_EQ(3, tuple.size());
}
