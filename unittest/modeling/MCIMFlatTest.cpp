#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/modeling/MCIM.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

TEST(FlatMCIM, indexesIterator)
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

  for (auto equation: eq)
    for (auto variable: var)
      expectedList.push_back(std::make_pair(equation, variable));

  size_t counter = 0;

  for (auto[equation, variable]: mcim.getIndexes()) {
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
  mcim.set({4, 2}, 0);
  mcim.set({4, 2}, 8);
  mcim.set({5, 3}, 0);
  mcim.set({5, 3}, 8);

  for (long i = 4; i < 6; ++i) {
    for (long j = 2; j < 4; ++j) {
      for (long k = 0; k < 9; ++k) {
        bool value = mcim.get({i, j}, k);

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
 * Sum of MCIMs.
 *
 * Input:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    1    0
 * (4,3)  1    0    1    0
 * (5,2)  0    1    0    1
 * (5,3)  1    0    0    1
 *
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  1    0    0    1
 * (4,3)  1    0    0    1
 * (5,2)  0    1    1    0
 * (5,3)  0    1    1    0
 *
 * Expected result:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  1    1    1    1
 * (4,3)  1    0    1    1
 * (5,2)  0    1    1    1
 * (5,3)  1    1    1    1
 */
TEST(FlatMCIM, sum)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim1(eq, var);
  mcim1.set({4, 2}, 1);
  mcim1.set({4, 2}, 2);
  mcim1.set({4, 3}, 0);
  mcim1.set({4, 3}, 2);
  mcim1.set({5, 2}, 1);
  mcim1.set({5, 2}, 3);
  mcim1.set({5, 3}, 0);
  mcim1.set({5, 3}, 3);

  MCIM mcim2(eq, var);
  mcim2.set({4, 2}, 0);
  mcim2.set({4, 2}, 3);
  mcim2.set({4, 3}, 0);
  mcim2.set({4, 3}, 3);
  mcim2.set({5, 2}, 1);
  mcim2.set({5, 2}, 2);
  mcim2.set({5, 3}, 1);
  mcim2.set({5, 3}, 2);

  MCIM result = mcim1 + mcim2;

  EXPECT_TRUE(result.get({4, 2}, 0));
  EXPECT_TRUE(result.get({4, 2}, 1));
  EXPECT_TRUE(result.get({4, 2}, 2));
  EXPECT_TRUE(result.get({4, 2}, 3));
  EXPECT_TRUE(result.get({4, 3}, 0));
  EXPECT_FALSE(result.get({4, 3}, 1));
  EXPECT_TRUE(result.get({4, 3}, 2));
  EXPECT_TRUE(result.get({4, 3}, 3));
  EXPECT_FALSE(result.get({5, 2}, 0));
  EXPECT_TRUE(result.get({5, 2}, 1));
  EXPECT_TRUE(result.get({5, 2}, 2));
  EXPECT_TRUE(result.get({5, 2}, 3));
  EXPECT_TRUE(result.get({5, 3}, 0));
  EXPECT_TRUE(result.get({5, 3}, 1));
  EXPECT_TRUE(result.get({5, 3}, 2));
  EXPECT_TRUE(result.get({5, 3}, 3));
}

/**
 * Difference of MCIMs.
 *
 * Input:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    1    0
 * (4,3)  1    0    1    0
 * (5,2)  0    1    0    1
 * (5,3)  1    0    0    1
 *
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  1    0    0    1
 * (4,3)  1    0    0    1
 * (5,2)  0    1    1    0
 * (5,3)  0    1    1    0
 *
 * Expected result:
 * 			 (0)  (1)  (2)  (3)
 * (4,2)  0    1    1    0
 * (4,3)  0    0    1    0
 * (5,2)  0    0    0    1
 * (5,3)  1    0    0    1
 */
TEST(FlatMCIM, difference)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim1(eq, var);
  mcim1.set({4, 2}, 1);
  mcim1.set({4, 2}, 2);
  mcim1.set({4, 3}, 0);
  mcim1.set({4, 3}, 2);
  mcim1.set({5, 2}, 1);
  mcim1.set({5, 2}, 3);
  mcim1.set({5, 3}, 0);
  mcim1.set({5, 3}, 3);

  MCIM mcim2(eq, var);
  mcim2.set({4, 2}, 0);
  mcim2.set({4, 2}, 3);
  mcim2.set({4, 3}, 0);
  mcim2.set({4, 3}, 3);
  mcim2.set({5, 2}, 1);
  mcim2.set({5, 2}, 2);
  mcim2.set({5, 3}, 1);
  mcim2.set({5, 3}, 2);

  MCIM result = mcim1 - mcim2;

  EXPECT_FALSE(result.get({4, 2}, 0));
  EXPECT_TRUE(result.get({4, 2}, 1));
  EXPECT_TRUE(result.get({4, 2}, 2));
  EXPECT_FALSE(result.get({4, 2}, 3));
  EXPECT_FALSE(result.get({4, 3}, 0));
  EXPECT_FALSE(result.get({4, 3}, 1));
  EXPECT_TRUE(result.get({4, 3}, 2));
  EXPECT_FALSE(result.get({4, 3}, 3));
  EXPECT_FALSE(result.get({5, 2}, 0));
  EXPECT_FALSE(result.get({5, 2}, 1));
  EXPECT_FALSE(result.get({5, 2}, 2));
  EXPECT_TRUE(result.get({5, 2}, 3));
  EXPECT_TRUE(result.get({5, 3}, 0));
  EXPECT_FALSE(result.get({5, 3}, 1));
  EXPECT_FALSE(result.get({5, 3}, 2));
  EXPECT_TRUE(result.get({5, 3}, 3));
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
TEST(FlatMCIM, flattenRows)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 9));

  MCIM mcim(eq, var);
  mcim.set({4, 2}, 1);
  mcim.set({4, 3}, 2);
  mcim.set({5, 2}, 3);
  mcim.set({5, 3}, 4);
  mcim.set({4, 2}, 5);
  mcim.set({4, 3}, 5);
  mcim.set({4, 2}, 6);
  mcim.set({5, 2}, 6);
  mcim.set({4, 2}, 7);
  mcim.set({5, 3}, 7);
  mcim.set({4, 2}, 8);
  mcim.set({4, 3}, 8);
  mcim.set({5, 2}, 8);
  mcim.set({5, 3}, 8);

  MCIS flattened = mcim.flattenRows();

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
TEST(FlatMCIM, flattenColumns)
{
  MultidimensionalRange eq({
      Range(4, 7),
      Range(1, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({4, 2}, 0);
  mcim.set({4, 3}, 1);
  mcim.set({5, 1}, 2);
  mcim.set({5, 2}, 3);
  mcim.set({5, 3}, 0);
  mcim.set({5, 3}, 1);
  mcim.set({6, 1}, 0);
  mcim.set({6, 1}, 2);
  mcim.set({6, 2}, 0);
  mcim.set({6, 2}, 3);
  mcim.set({6, 3}, 0);
  mcim.set({6, 3}, 1);
  mcim.set({6, 3}, 2);
  mcim.set({6, 3}, 3);

  MCIS flattened = mcim.flattenColumns();

  EXPECT_FALSE(flattened.contains({4, 1}));
  EXPECT_TRUE(flattened.contains({4, 2}));
  EXPECT_TRUE(flattened.contains({4, 3}));
  EXPECT_TRUE(flattened.contains({5, 1}));
  EXPECT_TRUE(flattened.contains({5, 2}));
  EXPECT_TRUE(flattened.contains({5, 3}));
  EXPECT_TRUE(flattened.contains({6, 1}));
  EXPECT_TRUE(flattened.contains({6, 2}));
  EXPECT_TRUE(flattened.contains({6, 3}));
}

/**
 * Filter a MCIM by row.
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
TEST(FlatMCIM, rowsFilter)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({4, 2}, 1);
  mcim.set({4, 2}, 3);
  mcim.set({4, 3}, 0);
  mcim.set({4, 3}, 2);
  mcim.set({5, 2}, 1);
  mcim.set({5, 2}, 3);
  mcim.set({5, 3}, 0);
  mcim.set({5, 3}, 2);

  MCIS filter;
  filter += {4, 2};
  filter += {5, 3};

  MCIM result = mcim.filterRows(filter);

  for (const auto&[equation, variable]: result.getIndexes()) {
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
 * Filter a MCIM by column.
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
TEST(FlatMCIM, columnsFilter)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  MCIM mcim(eq, var);
  mcim.set({4, 2}, 1);
  mcim.set({4, 2}, 3);
  mcim.set({4, 3}, 0);
  mcim.set({4, 3}, 2);
  mcim.set({5, 2}, 1);
  mcim.set({5, 2}, 3);
  mcim.set({5, 3}, 0);
  mcim.set({5, 3}, 2);

  MCIS filter;
  filter += 0;
  filter += 3;

  MCIM result = mcim.filterColumns(filter);

  for (const auto&[equation, variable]: result.getIndexes()) {
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
