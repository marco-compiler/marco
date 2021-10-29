#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "Common.h"

using namespace marco::matching;
using namespace marco::matching::detail;

/**
 * var
 * x[3]
 *
 * equ
 * e1 ; x[i] + x[i + 1] = 0 ; i[0:2)
 * e2 ; x[2] = 0
 */
TEST(Matching, testCase1)
{
	MatchingGraph<Variable, Equation> graph;

	Variable x("x", { 3 });
	graph.addVariable(x);

	Equation eq1("eq1");
	eq1.addIterationRange(Range(0, 2));
	eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
	eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 1)));
	graph.addEquation(eq1);

	Equation eq2("eq2");
	eq2.addIterationRange(Range(0, 1));
	eq2.addVariableAccess(Access(x, DimensionAccess::constant(2)));
	graph.addEquation(eq2);

	std::cout << "Equations: " << graph.getNumberOfScalarEquations() << std::endl;
	std::cout << "Variables: " << graph.getNumberOfScalarVariables() << std::endl;

	graph.simplify();
}

TEST(Matching, test1)
{
	MatchingGraph<Variable, Equation> graph;

	Variable x("x", { 3, 4 });
	graph.addVariable(x);

	Equation eq1("eq1");
	eq1.addIterationRange(Range(1, 4));
	eq1.addIterationRange(Range(2, 4));
	eq1.addVariableAccess(Access(x, DimensionAccess::relative(1, -1), DimensionAccess::constant(2)));
	graph.addEquation(eq1);

  graph.simplify();
}

TEST(Matching, test2)
{
	MatchingGraph<Variable, Equation> graph;

	Variable x("x", { 3, 4 });
	graph.addVariable(x);

	Equation eq1("eq1");
	eq1.addIterationRange(Range(1, 4));
	eq1.addIterationRange(Range(2, 4));
	eq1.addVariableAccess(Access(x, DimensionAccess::relative(1, -1), DimensionAccess::relative(1, -1)));
	graph.addEquation(eq1);

  graph.simplify();
}
