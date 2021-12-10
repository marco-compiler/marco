#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/Matching.h>

using namespace marco::matching;
using namespace marco::matching::detail;

/**
 * var x[3]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[i]
 *
 * Incidence matrix:
 * 			 (0)  (1)  (2)
 * (0,1)  1    0    0
 * (0,2)  1    0    0
 * (1,1)  0    1    0
 * (1,2)  0    1    0
 * (2,1)  0    0    1
 * (2,2)  0    0    1
 *
 * Expected solutions:
 * 			 (0)  (1)  (2)
 * (0,1)  1    0    0
 * (0,2)  0    0    0
 * (1,1)  0    1    0
 * (1,2)  0    0    0
 * (2,1)  0    0    1
 * (2,2)  0    0    0
 *
 * 			 (0)  (1)  (2)
 * (0,1)  0    0    0
 * (0,2)  1    0    0
 * (1,1)  0    0    0
 * (1,2)  0    1    0
 * (2,1)  0    0    0
 * (2,2)  0    0    1
 */
TEST(Matching, solveLocalMatchingProblem_2D_underdimensionedVariable_firstInductionVariable)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var(Range(0, 3));

	AccessFunction access(DimensionAccess::relative(0, 0));

	MCIM m0(eq, var);
	m0.set({0, 1, 0});
	m0.set({1, 1, 1});
	m0.set({2, 1, 2});

	MCIM m1(eq, var);
	m1.set({0, 2, 0});
	m1.set({1, 2, 1});
	m1.set({2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1));
}

/**
 * var x[3]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[j]
 *
 * Incidence matrix:
 * 			 (0)  (1)  (2)
 * (0,1)  0    1    0
 * (0,2)  0    0    1
 * (1,1)  0    1    0
 * (1,2)  0    0    1
 * (2,1)  0    1    0
 * (2,2)  0    0    1
 *
 * Expected solutions:
 * 			 (0)  (1)  (2)
 * (0,1)  0    1    0
 * (0,2)  0    0    1
 * (1,1)  0    0    0
 * (1,2)  0    0    0
 * (2,1)  0    0    0
 * (2,2)  0    0    0
 *
 * 			 (0)  (1)  (2)
 * (0,1)  0    0    0
 * (0,2)  0    0    0
 * (1,1)  0    1    0
 * (1,2)  0    0    1
 * (2,1)  0    0    0
 * (2,2)  0    0    0
 *
 * 			 (0)  (1)  (2)
 * (0,1)  0    0    0
 * (0,2)  0    0    0
 * (1,1)  0    0    0
 * (1,2)  0    0    0
 * (2,1)  0    1    0
 * (2,2)  0    0    1
 */
TEST(Matching, solveLocalMatchingProblem_2D_underdimensionedVariable_secondInductionVariable)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var(Range(0, 3));

	AccessFunction access(DimensionAccess::relative(1, 0));

  MCIM m0(eq, var);
	m0.set({0, 1, 1});
	m0.set({0, 2, 2});

  MCIM m1(eq, var);
	m1.set({1, 1, 1});
	m1.set({1, 2, 2});

  MCIM m2(eq, var);
	m2.set({2, 1, 1});
	m2.set({2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}

/**
 * var x[3][4]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[i][j]
 *
 * Incidence matrix:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)  0       1      0      0      0      0      0      0      0      0      0      0
 * (0,2)  0       0      1      0      0      0      0      0      0      0      0      0
 * (1,1)  0       0      0      0      0      1      0      0      0      0      0      0
 * (1,2)  0       0      0      0      0      0      1      0      0      0      0      0
 * (2,1)  0       0      0      0      0      0      0      0      0      1      0      0
 * (2,2)  0       0      0      0      0      0      0      0      0      0      1      0
 *
 * Expected solutions:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)  0       1      0      0      0      0      0      0      0      0      0      0
 * (0,2)  0       0      1      0      0      0      0      0      0      0      0      0
 * (1,1)  0       0      0      0      0      1      0      0      0      0      0      0
 * (1,2)  0       0      0      0      0      0      1      0      0      0      0      0
 * (2,1)  0       0      0      0      0      0      0      0      0      1      0      0
 * (2,2)  0       0      0      0      0      0      0      0      0      0      1      0
 */
TEST(Matching, solveLocalMatchingProblem_2D_allInductionVariablesUsed)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4)
	});

	AccessFunction access({
			DimensionAccess::relative(0, 0),
			DimensionAccess::relative(1, 0),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 0, 1});
	m0.set({0, 2, 0, 2});
	m0.set({1, 1, 1, 1});
	m0.set({1, 2, 1, 2});
	m0.set({2, 1, 2, 1});
	m0.set({2, 2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0));
}

/**
 * var x[3][4]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[j][i]
 *
 * Incidence matrix:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      1      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      1      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      1      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
 *
 * Expected solutions:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      1      0      0      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      1      0      0
 * (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
 */
TEST(Matching, solveLocalMatchingProblem_2D_invertedInductionVariables)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4)
	});

	AccessFunction access({
			DimensionAccess::relative(1, 0),
			DimensionAccess::relative(0, 0),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 1, 0});
	m0.set({0, 2, 2, 0});
	m0.set({1, 1, 1, 1});
	m0.set({1, 2, 2, 1});
	m0.set({2, 1, 1, 2});
	m0.set({2, 2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0));
}

/**
 * var x[3][4]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[2][i]
 *
 * Incidence matrix:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      1      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      1      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
 *
 * Expected solutions:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      1      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      1      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
 */
TEST(Matching, solveLocalMatchingProblem_2D_oneConstantIndex)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4)
	});

	AccessFunction access({
			DimensionAccess::constant(1),
			DimensionAccess::relative(0, 0),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 1, 0});
	m0.set({1, 1, 1, 1});
	m0.set({2, 1, 1, 2});

  MCIM m1(eq, var);
	m1.set({0, 2, 1, 0});
	m1.set({1, 2, 1, 1});
	m1.set({2, 2, 1, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1));
}

/**
 * var x[3][4]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[1][2]
 *
 * Incidence matrix:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      1      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
 *
 * Expected solutions:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      1      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
 */
TEST(Matching, solveLocalMatchingProblem_2D_allConstantIndexes)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4)
	});

	AccessFunction access({
			DimensionAccess::constant(1),
			DimensionAccess::constant(2),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 1, 2});

  MCIM m1(eq, var);
	m1.set({0, 2, 1, 2});

  MCIM m2(eq, var);
	m2.set({1, 1, 1, 2});

  MCIM m3(eq, var);
	m3.set({1, 2, 1, 2});

  MCIM m4(eq, var);
	m4.set({2, 1, 1, 2});

  MCIM m5(eq, var);
	m5.set({2, 2, 1, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2, m3, m4, m5));
}

/**
 * var x[3][4]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   x[j][j]
 *
 * Incidence matrix:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      1      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      1      0
 * (2,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
 *
 * Expected solutions:
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      1      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      1      0
 * (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
 *
 * 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
 * (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
 * (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
 * (2,1)   0      0      0      0      0      1      0      0      0      0      0      0
 * (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
 */
TEST(Matching, solveLocalMatchingProblem_2D_repeatedInductionVariable)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4)
	});

	AccessFunction access({
			DimensionAccess::relative(1, 0),
			DimensionAccess::relative(1, 0),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 1, 1});
	m0.set({0, 2, 2, 2});

  MCIM m1(eq, var);
	m1.set({1, 1, 1, 1});
	m1.set({1, 2, 2, 2});

  MCIM m2(eq, var);
	m2.set({2, 1, 1, 1});
	m2.set({2, 2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}


/**
 * var x[3][4][5]
 *
 * for i in [0:3)
 * 	 for j in [1,3)
 * 	   for k in [1,4)
 * 	     x[j][j][i]
 */
TEST(Matching, solveLocalMatchingProblem_3D_repeatedInductionVariable)
{
	MultidimensionalRange eq({
			Range(0, 3),
			Range(1, 3),
			Range(1, 4)
	});

	MultidimensionalRange var({
			Range(0, 3),
			Range(0, 4),
			Range(0, 5)
	});

	AccessFunction access({
			DimensionAccess::relative(1, 0),
			DimensionAccess::relative(1, 0),
			DimensionAccess::relative(0, 0),
	});

  MCIM m0(eq, var);
	m0.set({0, 1, 1, 1, 1, 0});
	m0.set({0, 2, 1, 2, 2, 0});
	m0.set({1, 1, 1, 1, 1, 1});
	m0.set({1, 2, 1, 2, 2, 1});
	m0.set({2, 1, 1, 1, 1, 2});
	m0.set({2, 2, 1, 2, 2, 2});

  MCIM m1(eq, var);
	m1.set({0, 1, 2, 1, 1, 0});
	m1.set({0, 2, 2, 2, 2, 0});
	m1.set({1, 1, 2, 1, 1, 1});
	m1.set({1, 2, 2, 2, 2, 1});
	m1.set({2, 1, 2, 1, 1, 2});
	m1.set({2, 2, 2, 2, 2, 2});

  MCIM m2(eq, var);
	m2.set({0, 1, 3, 1, 1, 0});
	m2.set({0, 2, 3, 2, 2, 0});
	m2.set({1, 1, 3, 1, 1, 1});
	m2.set({1, 2, 3, 2, 2, 1});
	m2.set({2, 1, 3, 1, 1, 2});
	m2.set({2, 2, 3, 2, 2, 2});

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(eq, var, access))
    solutions.push_back(solution);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}