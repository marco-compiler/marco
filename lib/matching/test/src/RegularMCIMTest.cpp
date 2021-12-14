#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/MCIM.h>

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(RegularMCIM, indexesIterator)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 3)
  });

  MCIM mcim(eq, var);

  llvm::SmallVector<std::pair<Point, Point>, 8> expectedList;

  for (auto equation : eq)
    for (auto variable : var)
      expectedList.push_back(std::make_pair(equation, variable));

  size_t counter = 0;

  for (auto [equation, variable] : mcim.getIndexes())
  {
    ASSERT_LT(counter, expectedList.size());

    auto& expectedEquation = expectedList[counter].first;
    EXPECT_EQ(equation, expectedEquation);

    auto& expectedVariable = expectedList[counter].second;
    EXPECT_EQ(variable, expectedVariable);

    ++counter;
  }

  EXPECT_EQ(counter, eq.flatSize() * var.flatSize());
}

/**
 * Try setting to true the 4 vertices of the matrix.
 *
 * 			 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)
 * (4,2)   1      0      0      0      0      0      0      1
 * (4,3)   0      0      0      0      0      0      0      0
 * (5,2)   0      0      0      0      0      0      0      0
 * (5,3)   1      0      0      0      0      0      0      1
 */
TEST(RegularMCIM, set)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 4),
    Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 0, 0 });
  mcim.set({ 4, 2 }, { 3, 1 });
  mcim.set({ 5, 3 }, {0, 0 });
  mcim.set({ 5, 3 }, {3, 1 });

  for (long i = 4; i < 6; ++i)
  {
    for (long j = 2; j < 4; ++j)
    {
      for (long k = 0; k < 4; ++k)
      {
        for (long l = 0; l < 2; ++l)
        {
          bool value = mcim.get({ i, j }, { k, l });

          if (i == 4 && j == 2 && k == 0 && l == 0)
            EXPECT_TRUE(value);
          else if (i == 4 && j == 2 && k == 3 && l == 1)
            EXPECT_TRUE(value);
          else if (i == 5 && j == 3 && k == 0 && l == 0)
            EXPECT_TRUE(value);
          else if (i == 5 && j == 3 && k == 3 && l == 1)
            EXPECT_TRUE(value);
          else
            EXPECT_FALSE(value);
        }
      }
    }
  }
}

/**
 * Sum of MCIMs.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      1      0
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      0      1
 *
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   1      0      0      1
 * (4,3)   1      0      0      1
 * (5,2)   0      1      1      0
 * (5,3)   0      1      1      0
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   1      1      1      1
 * (4,3)   1      0      1      1
 * (5,2)   0      1      1      1
 * (5,3)   1      1      1      1
 */
TEST(RegularMCIM, sum)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  MCIM mcim1(eq, var);
  mcim1.set({ 4, 2 }, { 0, 1 });
  mcim1.set({ 4, 2 }, { 1, 0 });
  mcim1.set({ 4, 3 }, { 0, 0 });
  mcim1.set({ 4, 3 }, { 1, 0 });
  mcim1.set({ 5, 2 }, { 0, 1 });
  mcim1.set({ 5, 2 }, { 1, 1 });
  mcim1.set({ 5, 3 }, { 0, 0 });
  mcim1.set({ 5, 3 }, { 1, 1 });

  MCIM mcim2(eq, var);
  mcim2.set({ 4, 2 }, { 0, 0 });
  mcim2.set({ 4, 2 }, { 1, 1 });
  mcim2.set({ 4, 3 }, { 0, 0 });
  mcim2.set({ 4, 3 }, { 1, 1 });
  mcim2.set({ 5, 2 }, { 0, 1 });
  mcim2.set({ 5, 2 }, { 1, 0 });
  mcim2.set({ 5, 3 }, { 0, 1 });
  mcim2.set({ 5, 3 }, { 1, 0 });

  MCIM result = mcim1 + mcim2;

  EXPECT_TRUE(result.get({ 4, 2 }, { 0, 0 }));
  EXPECT_TRUE(result.get({ 4, 2 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 4, 2 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 4, 2 }, { 1, 1 }));
  EXPECT_TRUE(result.get({ 4, 3 }, { 0, 0 }));
  EXPECT_FALSE(result.get({ 4, 3 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 4, 3 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 4, 3 }, { 1, 1 }));
  EXPECT_FALSE(result.get({ 5, 2 }, { 0, 0 }));
  EXPECT_TRUE(result.get({ 5, 2 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 5, 2 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 5, 2 }, { 1, 1 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 0, 0 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 1, 1 }));
}

/**
 * Difference of MCIMs.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      1      0
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      0      1
 *
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   1      0      0      1
 * (4,3)   1      0      0      1
 * (5,2)   0      1      1      0
 * (5,3)   0      1      1      0
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      1      0
 * (4,3)   0      0      1      0
 * (5,2)   0      0      0      1
 * (5,3)   1      0      0      1
 */
TEST(RegularMCIM, difference)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  MCIM mcim1(eq, var);
  mcim1.set({ 4, 2 }, { 0, 1 });
  mcim1.set({ 4, 2 }, { 1, 0 });
  mcim1.set({ 4, 3 }, { 0, 0 });
  mcim1.set({ 4, 3 }, { 1, 0 });
  mcim1.set({ 5, 2 }, { 0, 1 });
  mcim1.set({ 5, 2 }, { 1, 1 });
  mcim1.set({ 5, 3 }, { 0, 0 });
  mcim1.set({ 5, 3 }, { 1, 1 });

  MCIM mcim2(eq, var);
  mcim2.set({ 4, 2 }, { 0, 0 });
  mcim2.set({ 4, 2 }, { 1, 1 });
  mcim2.set({ 4, 3 }, { 0, 0 });
  mcim2.set({ 4, 3 }, { 1, 1 });
  mcim2.set({ 5, 2 }, { 0, 1 });
  mcim2.set({ 5, 2 }, { 1, 0 });
  mcim2.set({ 5, 3 }, { 0, 1 });
  mcim2.set({ 5, 3 }, { 1, 0 });

  MCIM result = mcim1 - mcim2;

  EXPECT_FALSE(result.get({ 4, 2 }, { 0, 0 }));
  EXPECT_TRUE(result.get({ 4, 2 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 4, 2 }, { 1, 0 }));
  EXPECT_FALSE(result.get({ 4, 2 }, { 1, 1 }));
  EXPECT_FALSE(result.get({ 4, 3 }, { 0, 0 }));
  EXPECT_FALSE(result.get({ 4, 3 }, { 0, 1 }));
  EXPECT_TRUE(result.get({ 4, 3 }, { 1, 0 }));
  EXPECT_FALSE(result.get({ 4, 3 }, { 1, 1 }));
  EXPECT_FALSE(result.get({ 5, 2 }, { 0, 0 }));
  EXPECT_FALSE(result.get({ 5, 2 }, { 0, 1 }));
  EXPECT_FALSE(result.get({ 5, 2 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 5, 2 }, { 1, 1 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 0, 0 }));
  EXPECT_FALSE(result.get({ 5, 3 }, { 0, 1 }));
  EXPECT_FALSE(result.get({ 5, 3 }, { 1, 0 }));
  EXPECT_TRUE(result.get({ 5, 3 }, { 1, 1 }));
}

/**
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)  (4,0)  (4,1)
 * (4,2)   0      1      0      0      0      0      1      1      1      1
 * (4,3)   0      0      1      0      0      0      1      0      0      1
 * (5,2)   0      0      0      1      0      0      0      1      0      1
 * (5,3)   0      0      0      0      1      0      0      0      1      1
 *
 * Expected result:
 * {(0,1), (1,0), (1,1), (2,0), (3,0), (3,1), (4,0), (4,1)}
 */
TEST(RegularMCIM, flattenEquations)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 5),
    Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 0, 1 });
  mcim.set({ 4, 2 }, { 3, 0 });
  mcim.set({ 4, 2 }, { 3, 1 });
  mcim.set({ 4, 2 }, { 4, 0 });
  mcim.set({ 4, 2 }, { 4, 1 });
  mcim.set({ 4, 3 }, { 1, 0 });
  mcim.set({ 4, 3 }, { 3, 0 });
  mcim.set({ 4, 3 }, { 4, 1 });
  mcim.set({ 5, 2 }, { 1, 1 });
  mcim.set({ 5, 2 }, { 3, 1 });
  mcim.set({ 5, 2 }, { 4, 1 });
  mcim.set({ 5, 3 }, { 2, 0 });
  mcim.set({ 5, 3 }, { 4, 0 });
  mcim.set({ 5, 3 }, { 4, 1 });

  MCIS flattened = mcim.flattenEquations();

  EXPECT_FALSE(flattened.contains({ 0, 0 }));
  EXPECT_TRUE(flattened.contains({ 0, 1 }));
  EXPECT_TRUE(flattened.contains({ 1, 0 }));
  EXPECT_TRUE(flattened.contains({ 1, 1 }));
  EXPECT_TRUE(flattened.contains({ 2, 0 }));
  EXPECT_FALSE(flattened.contains({ 2, 1 }));
  EXPECT_TRUE(flattened.contains({ 3, 0 }));
  EXPECT_TRUE(flattened.contains({ 3, 1 }));
  EXPECT_TRUE(flattened.contains({ 4, 0 }));
  EXPECT_TRUE(flattened.contains({ 4, 1 }));
}

/**
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,1)   0      0      0      0
 * (4,2)   1      0      0      0
 * (4,3)   0      1      0      0
 * (5,1)   0      0      1      0
 * (5,2)   0      0      0      1
 * (5,3)   1      1      0      0
 * (6,1)   1      0      1      0
 * (6,2)   1      0      0      1
 * (6,3)   1      1      1      1
 *
 * Expected result:
 * {(4,1), (4,2), (4,3), (5,1), (5,2), (5,3), (6,1), (6,2), (6,3)}
 */
TEST(RegularMCIM, flattenVariables)
{
  MultidimensionalRange eq({
    Range(4, 7),
    Range(1, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 0, 0 });
  mcim.set({ 4, 3 }, { 0, 1 });
  mcim.set({ 5, 1 }, { 1, 0 });
  mcim.set({ 5, 2 }, { 1, 1 });
  mcim.set({ 5, 3 }, { 0, 0 });
  mcim.set({ 5, 3 }, { 0, 1 });
  mcim.set({ 6, 1 }, { 0, 0 });
  mcim.set({ 6, 1 }, { 1, 0 });
  mcim.set({ 6, 2 }, { 0, 0 });
  mcim.set({ 6, 2 }, { 1, 1 });
  mcim.set({ 6, 3 }, { 0, 0 });
  mcim.set({ 6, 3 }, { 0, 1 });
  mcim.set({ 6, 3 }, { 1, 0 });
  mcim.set({ 6, 3 }, { 1, 1 });

  MCIS flattened = mcim.flattenVariables();

  EXPECT_FALSE(flattened.contains({ 4, 1 }));
  EXPECT_TRUE(flattened.contains({ 4, 2 }));
  EXPECT_TRUE(flattened.contains({ 4, 3 }));
  EXPECT_TRUE(flattened.contains({ 5, 1 }));
  EXPECT_TRUE(flattened.contains({ 5, 2 }));
  EXPECT_TRUE(flattened.contains({ 5, 3 }));
  EXPECT_TRUE(flattened.contains({ 6, 1 }));
  EXPECT_TRUE(flattened.contains({ 6, 2 }));
  EXPECT_TRUE(flattened.contains({ 6, 3 }));
}

/**
 * Filter a MCIM by equation.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      1      0
 *
 * {(4,2), (5,3)}
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   0      0      0      0
 * (5,2)   0      0      0      0
 * (5,3)   1      0      1      0
 */
TEST(RegularMCIM, equationsFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 0, 1 });
  mcim.set({ 4, 2 }, { 1, 1 });
  mcim.set({ 4, 3 }, { 0, 0 });
  mcim.set({ 4, 3 }, { 1, 0 });
  mcim.set({ 5, 2 }, { 0, 1 });
  mcim.set({ 5, 2 }, { 1, 1 });
  mcim.set({ 5, 3 }, { 0, 0 });
  mcim.set({ 5, 3 }, { 1, 0 });

  MultidimensionalRange filterEq({
    Range(4, 6),
    Range(2, 4)
  });

  MCIS filter;
  filter += { 4, 2 };
  filter += { 5, 3 };

  MCIM result = mcim.filterEquations(filter);

  for (const auto& [equation, variable] : result.getIndexes())
  {
    bool value = result.get(equation, variable);

    if (equation[0] == 4 && equation[1] == 2 && variable[0] == 0 && variable[1] == 1)
      EXPECT_TRUE(value);
    else if (equation[0] == 4 && equation[1] == 2 && variable[0] == 1 && variable[1] == 1)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 0 && variable[1] == 0)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 1 && variable[1] == 0)
      EXPECT_TRUE(value);
    else
      EXPECT_FALSE(value);
  }
}

/**
 * Filter a MCIM by variable.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      1      0
 *
 * {(0,0), (1,1)}
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      0      0      1
 * (4,3)   1      0      0      0
 * (5,2)   0      0      0      1
 * (5,3)   1      0      0      0
 */
TEST(RegularMCIM, variablesFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 0, 1 });
  mcim.set({ 4, 2 }, { 1, 1 });
  mcim.set({ 4, 3 }, { 0, 0 });
  mcim.set({ 4, 3 }, { 1, 0 });
  mcim.set({ 5, 2 }, { 0, 1 });
  mcim.set({ 5, 2 }, { 1, 1 });
  mcim.set({ 5, 3 }, { 0, 0 });
  mcim.set({ 5, 3 }, { 1, 0 });

  MultidimensionalRange filterEq({
    Range(4, 6),
    Range(2, 4)
  });

  MCIS filter;
  filter += { 0, 0 };
  filter += { 1, 1 };

  MCIM result = mcim.filterVariables(filter);

  for (const auto& [equation, variable] : result.getIndexes())
  {
    bool value = result.get(equation, variable);

    if (equation[0] == 4 && equation[1] == 2 && variable[0] == 1 && variable[1] == 1)
      EXPECT_TRUE(value);
    else if (equation[0] == 4 && equation[1] == 3 && variable[0] == 0 && variable[1] == 0)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 2 && variable[0] == 1 && variable[1] == 1)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 0 && variable[1] == 0)
      EXPECT_TRUE(value);
    else
      EXPECT_FALSE(value);
  }
}