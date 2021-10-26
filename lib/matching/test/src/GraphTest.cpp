#include <gtest/gtest.h>
#include <marco/matching/Graph.h>

#include "Common.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, singleVariableInsertion)
{
	MatchingGraph<Variable, Equation> graph;
	graph.addVariable(Variable("x"));
	EXPECT_TRUE(graph.hasVariable("x"));
}

TEST(Matching, multipleVariablesInsertion)
{
	MatchingGraph<Variable, Equation> graph;

	graph.addVariable(Variable("x"));
	graph.addVariable(Variable("y"));

	EXPECT_TRUE(graph.hasVariable("x"));
	EXPECT_TRUE(graph.hasVariable("y"));
	EXPECT_FALSE(graph.hasVariable("z"));
}

TEST(Matching, multidimensionalVariableInsertion)
{
	MatchingGraph<Variable, Equation> graph;
	graph.addVariable(Variable("x", { 2, 3, 4 }));

	auto var = graph.getVariable("x");
	EXPECT_EQ(var.getRank(), 3);
	EXPECT_EQ(var.getDimensionSize(0), 2);
	EXPECT_EQ(var.getDimensionSize(1), 3);
	EXPECT_EQ(var.getDimensionSize(2), 4);
}

TEST(Matching, addEquation)
{
	MatchingGraph<Variable, Equation> graph;

	Variable x("x", { 2 });

	Equation eq1("eq1");
	eq1.addIterationRange(Range(0, 1));
	eq1.addVariableAccess(Access(x, DimensionAccess::constant(0)));
	eq1.addVariableAccess(Access(x, DimensionAccess::constant(1)));

	graph.addVariable(x);
	graph.addEquation(eq1);

	ASSERT_TRUE(graph.hasEquation("eq1"));
	ASSERT_TRUE(graph.hasEdge("eq1", "x"));
}
