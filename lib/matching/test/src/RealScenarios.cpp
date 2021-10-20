#include <gtest/gtest.h>
#include <marco/matching/Graph.h>

#include "Common.h"

using namespace marco::matching;

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
	eq1.addVariableAccess(Access(x, SingleDimensionAccess::relative(0, 0)));
	eq1.addVariableAccess(Access(x, SingleDimensionAccess::relative(0, 1)));
	graph.addEquation(eq1);

	Equation eq2("eq2");
	eq2.addIterationRange(Range(0, 1));
	eq2.addVariableAccess(Access(x, SingleDimensionAccess::constant(2)));
	graph.addEquation(eq2);

	std::cout << "Equations: " << graph.getNumberOfScalarEquations() << std::endl;
	std::cout << "Variables: " << graph.getNumberOfScalarVariables() << std::endl;
}
