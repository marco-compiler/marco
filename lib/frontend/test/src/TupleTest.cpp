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
	SourcePosition location("-", 0, 0);
	Expression exp(location, Type::Int(), ReferenceAccess("x"));
	Tuple tuple({ exp });
	EXPECT_EQ(1, tuple.size());
}

TEST(TupleTest, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	SourcePosition location("-", 0, 0);
	Expression exp1(location, Type::Int(), ReferenceAccess("x"));
	Expression exp2(location, Type::Float(), ReferenceAccess("y"));
	Expression exp3(location, Type::unknown(), ReferenceAccess("z"));
	Tuple tuple({ exp1, exp2, exp3 });
	EXPECT_EQ(3, tuple.size());
}
