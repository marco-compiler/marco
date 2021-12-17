#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/MCIM.h>

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(MCIM, indexesIterator)
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
 * Try setting to true the 4 vertices of the matrix.
 *
 * 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
 * (4,2)  1    0    0    0    0    0    0    0    1
 * (4,3)  0    0    0    0    0    0    0    0    0
 * (5,2)  0    0    0    0    0    0    0    0    0
 * (5,3)  1    0    0    0    0    0    0    0    1
 */
TEST(FlatMCIM, set)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 9));

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, 0);
  mcim.set({ 4, 2 }, 8);
  mcim.set({ 5, 3 }, 0);
  mcim.set({ 5, 3 }, 8);

  for (long i = 4; i < 6; ++i)
  {
    for (long j = 2; j < 4; ++j)
    {
      for (long k = 0; k < 9; ++k)
      {
        bool value = mcim.get({ i, j }, k);

        if (i == 4 && j == 2 && k == 0)
          EXPECT_TRUE(value);
        else if (i == 4 && j == 2 && k == 8)
          EXPECT_TRUE(value);
        else if (i == 5 && j == 3 && k == 0)
          EXPECT_TRUE(value);
        else if (i == 5 && j == 3 && k == 8)
          EXPECT_TRUE(value);
        else
          EXPECT_FALSE(value);
      }
    }
  }
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
 * 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
 * (4,2)  0    1    0    0    0    1    1    1    1
 * (4,3)  0    0    1    0    0    1    0    0    1
 * (5,2)  0    0    0    1    0    0    1    0    1
 * (5,3)  0    0    0    0    1    0    0    1    1
 *
 * Expected result:
 * {1, 2, 3, 4, 5, 6, 7, 8}
 */
TEST(FlatMCIM, flattenEquations)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 9));

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, { 1 });
  mcim.set({ 4, 3 }, { 2 });
  mcim.set({ 5, 2 }, { 3 });
  mcim.set({ 5, 3 }, { 4 });
  mcim.set({ 4, 2 }, { 5 });
  mcim.set({ 4, 3 }, { 5 });
  mcim.set({ 4, 2 }, { 6 });
  mcim.set({ 5, 2 }, { 6 });
  mcim.set({ 4, 2 }, { 7 });
  mcim.set({ 5, 3 }, { 7 });
  mcim.set({ 4, 2 }, { 8 });
  mcim.set({ 4, 3 }, { 8 });
  mcim.set({ 5, 2 }, { 8 });
  mcim.set({ 5, 3 }, { 8 });

  MCIS flattened = mcim.flattenEquations();
  EXPECT_FALSE(flattened.contains(0));
  EXPECT_TRUE(flattened.contains(1));
  EXPECT_TRUE(flattened.contains(2));
  EXPECT_TRUE(flattened.contains(3));
  EXPECT_TRUE(flattened.contains(4));
  EXPECT_TRUE(flattened.contains(5));
  EXPECT_TRUE(flattened.contains(6));
  EXPECT_TRUE(flattened.contains(7));
  EXPECT_TRUE(flattened.contains(8));
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
 * Input:
 * 			 (0)  (1)  (2)  (3)
 * (4,1)  0    0    0    0
 * (4,2)  1    0    0    0
 * (4,3)  0    1    0    0
 * (5,1)  0    0    1    0
 * (5,2)  0    0    0    1
 * (5,3)  1    1    0    0
 * (6,1)  1    0    1    0
 * (6,2)  1    0    0    1
 * (6,3)  1    1    1    1
 *
 * Expected result:
 * {(4,2), (4,3), (5,1), (5,2), (5,3), (6,1), (6,2), (6,3)}
 */
TEST(FlatMCIM, flattenVariables)
{
  MultidimensionalRange eq({
    Range(4, 7),
    Range(1, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, 0);
  mcim.set({ 4, 3 }, 1);
  mcim.set({ 5, 1 }, 2);
  mcim.set({ 5, 2 }, 3);
  mcim.set({ 5, 3 }, 0);
  mcim.set({ 5, 3 }, 1);
  mcim.set({ 6, 1 }, 0);
  mcim.set({ 6, 1 }, 2);
  mcim.set({ 6, 2 }, 0);
  mcim.set({ 6, 2 }, 3);
  mcim.set({ 6, 3 }, 0);
  mcim.set({ 6, 3 }, 1);
  mcim.set({ 6, 3 }, 2);
  mcim.set({ 6, 3 }, 3);

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
 * Filter a MCIM by equation.
 *
 * Input:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    0    1
 * (4,3)  1    0    1    0
 * (5,2)  0    1    0    1
 * (5,3)  1    0    1    0
 *
 * {(4,2), (5,3)}
 *
 * Expected result:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    0    1
 * (4,3)  0    0    0    0
 * (5,2)  0    0    0    0
 * (5,3)  1    0    1    0
 */
TEST(FlatMCIM, equationsFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, 1);
  mcim.set({ 4, 2 }, 3);
  mcim.set({ 4, 3 }, 0);
  mcim.set({ 4, 3 }, 2);
  mcim.set({ 5, 2 }, 1);
  mcim.set({ 5, 2 }, 3);
  mcim.set({ 5, 3 }, 0);
  mcim.set({ 5, 3 }, 2);

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

    if (equation[0] == 4 && equation[1] == 2 && variable[0] == 1)
      EXPECT_TRUE(value);
    else if (equation[0] == 4 && equation[1] == 2 && variable[0] == 3)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 0)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 2)
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

/**
 * Filter a MCIM by variable.
 *
 * Input:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    0    1
 * (4,3)  1    0    1    0
 * (5,2)  0    1    0    1
 * (5,3)  1    0    1    0
 *
 * {0, 3}
 *
 * Expected result:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    0    0    1
 * (4,3)  1    0    0    0
 * (5,2)  0    0    0    1
 * (5,3)  1    0    0    0
 */
TEST(FlatMCIM, variablesFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({ 4, 2 }, 1);
  mcim.set({ 4, 2 }, 3);
  mcim.set({ 4, 3 }, 0);
  mcim.set({ 4, 3 }, 2);
  mcim.set({ 5, 2 }, 1);
  mcim.set({ 5, 2 }, 3);
  mcim.set({ 5, 3 }, 0);
  mcim.set({ 5, 3 }, 2);

  MCIS filter;
  filter += 0;
  filter += 3;

  MCIM result = mcim.filterVariables(filter);

  for (const auto& [equation, variable] : result.getIndexes())
  {
    bool value = result.get(equation, variable);

    if (equation[0] == 4 && equation[1] == 2 && variable[0] == 3)
      EXPECT_TRUE(value);
    else if (equation[0] == 4 && equation[1] == 3 && variable[0] == 0)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 2 && variable[0] == 3)
      EXPECT_TRUE(value);
    else if (equation[0] == 5 && equation[1] == 3 && variable[0] == 0)
      EXPECT_TRUE(value);
    else
      EXPECT_FALSE(value);
  }
}
