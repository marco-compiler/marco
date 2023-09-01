#include "marco/Modeling/MCIM.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

//===----------------------------------------------------------------------===//
// Equally dimensioned equations and variables
//===----------------------------------------------------------------------===//

TEST(MCIM_SameRank, indexesIterator)
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

  std::vector<std::pair<Point, Point>> actual;

  for (auto [equation, variable] : llvm::make_range(mcim.indicesBegin(), mcim.indicesEnd())) {
    actual.emplace_back(equation, variable);
  }

  std::vector<std::pair<Point, Point>> expected;

  for (auto equation : eq) {
    for (auto variable : var) {
      expected.push_back(std::make_pair(equation, variable));
    }
  }

  ASSERT_THAT(actual, testing::UnorderedElementsAreArray(expected.begin(), expected.end()));
}

//===----------------------------------------------------------------------===//
// Equally dimensioned equations and variables, but ragged
//===----------------------------------------------------------------------===//

TEST(MCIM_SameRank, indexesIteratorRagged)
{
  // IndexSet representing the indexes of an array of shape [2,{2,3}]
  IndexSet eq({  
    MultidimensionalRange({
      Range(0, 3),
      Range(0, 3)
    }),
    MultidimensionalRange({
      Range(0, 3),
      Range(0, 4)
    }),
  });

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 3)
  });

  MCIM mcim(eq, IndexSet(var));

  std::vector<std::pair<Point, Point>> actual;

  for (auto [equation, variable] : llvm::make_range(mcim.indicesBegin(), mcim.indicesEnd())) {
    actual.emplace_back(equation, variable);
  }

  std::vector<std::pair<Point, Point>> expected;

  for (auto equation : eq) {
    for (auto variable : var) {
      expected.push_back(std::make_pair(equation, variable));
    }
  }

  ASSERT_THAT(actual, testing::UnorderedElementsAreArray(expected.begin(), expected.end()));
}

// Try setting to true the 4 vertices of the matrix.
//
// 			 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)
// (4,2)   1      0      0      0      0      0      0      1
// (4,3)   0      0      0      0      0      0      0      0
// (5,2)   0      0      0      0      0      0      0      0
// (5,3)   1      0      0      0      0      0      0      1

TEST(MCIM_SameRank, set)
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
  mcim.set({4, 2}, {0, 0});
  mcim.set({4, 2}, {3, 1});
  mcim.set({5, 3}, {0, 0});
  mcim.set({5, 3}, {3, 1});

  EXPECT_TRUE(mcim.get({4, 2}, {0, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {0, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {1, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {1, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {2, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {2, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {3, 0}));
  EXPECT_TRUE(mcim.get({4, 2}, {3, 1}));

  EXPECT_FALSE(mcim.get({4, 3}, {0, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {0, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {1, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {1, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {2, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {2, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {3, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {3, 1}));

  EXPECT_FALSE(mcim.get({5, 2}, {0, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {0, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {1, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {1, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {2, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {2, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {3, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {3, 1}));

  EXPECT_TRUE(mcim.get({5, 3}, {0, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {0, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {1, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {1, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {2, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {2, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {3, 0}));
  EXPECT_TRUE(mcim.get({5, 3}, {3, 1}));
}

// Try setting to true the 4 vertices of the matrix.
//
// 			 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)
// (4,2)   1      0      0      0      0      0      0      1
// (4,3)   0      0      0      0      0      0      0      0
// (5,2)   0      0      0      0      0      0      0      0
// (5,3)   1      0      0      0      0      0      0      1
// (5,4)   0      0      0      0      0      0      0      1
TEST(MCIM_SameRank, setRagged)
{

  IndexSet eq({  
    MultidimensionalRange({
      Range(4, 6),
      Range(2, 4)
    }),
    MultidimensionalRange({
      Range(5, 6),
      Range(4, 5)
    }),
  });

  MultidimensionalRange var({
      Range(0, 4),
      Range(0, 2)
  });

  MCIM mcim(eq, IndexSet(var));
  mcim.set({4, 2}, {0, 0});
  mcim.set({4, 2}, {3, 1});
  mcim.set({5, 3}, {0, 0});
  mcim.set({5, 3}, {3, 1});
  mcim.set({5, 4}, {2, 1});

  EXPECT_TRUE(mcim.get({4, 2}, {0, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {0, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {1, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {1, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {2, 0}));
  EXPECT_FALSE(mcim.get({4, 2}, {2, 1}));
  EXPECT_FALSE(mcim.get({4, 2}, {3, 0}));
  EXPECT_TRUE(mcim.get({4, 2}, {3, 1}));

  EXPECT_FALSE(mcim.get({4, 3}, {0, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {0, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {1, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {1, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {2, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {2, 1}));
  EXPECT_FALSE(mcim.get({4, 3}, {3, 0}));
  EXPECT_FALSE(mcim.get({4, 3}, {3, 1}));

  EXPECT_FALSE(mcim.get({5, 2}, {0, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {0, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {1, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {1, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {2, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {2, 1}));
  EXPECT_FALSE(mcim.get({5, 2}, {3, 0}));
  EXPECT_FALSE(mcim.get({5, 2}, {3, 1}));

  EXPECT_TRUE(mcim.get({5, 3}, {0, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {0, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {1, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {1, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {2, 0}));
  EXPECT_FALSE(mcim.get({5, 3}, {2, 1}));
  EXPECT_FALSE(mcim.get({5, 3}, {3, 0}));
  EXPECT_TRUE(mcim.get({5, 3}, {3, 1}));

  EXPECT_FALSE(mcim.get({5, 4}, {0, 0}));
  EXPECT_FALSE(mcim.get({5, 4}, {0, 1}));
  EXPECT_FALSE(mcim.get({5, 4}, {1, 0}));
  EXPECT_FALSE(mcim.get({5, 4}, {1, 1}));
  EXPECT_FALSE(mcim.get({5, 4}, {2, 0}));
  EXPECT_TRUE(mcim.get({5, 4}, {2, 1}));
  EXPECT_FALSE(mcim.get({5, 4}, {3, 0}));
  EXPECT_FALSE(mcim.get({5, 4}, {3, 1}));
}

// Sum of MCIMs.
//
// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      1      0
// (4,3)   1      0      1      0
// (5,2)   0      1      0      1
// (5,3)   1      0      0      1
//
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   1      0      0      1
// (4,3)   1      0      0      1
// (5,2)   0      1      1      0
// (5,3)   0      1      1      0
//
// Expected result:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   1      1      1      1
// (4,3)   1      0      1      1
// (5,2)   0      1      1      1
// (5,3)   1      1      1      1

TEST(MCIM_SameRank, sum)
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
  mcim1.set({4, 2}, {0, 1});
  mcim1.set({4, 2}, {1, 0});
  mcim1.set({4, 3}, {0, 0});
  mcim1.set({4, 3}, {1, 0});
  mcim1.set({5, 2}, {0, 1});
  mcim1.set({5, 2}, {1, 1});
  mcim1.set({5, 3}, {0, 0});
  mcim1.set({5, 3}, {1, 1});

  MCIM mcim2(eq, var);
  mcim2.set({4, 2}, {0, 0});
  mcim2.set({4, 2}, {1, 1});
  mcim2.set({4, 3}, {0, 0});
  mcim2.set({4, 3}, {1, 1});
  mcim2.set({5, 2}, {0, 1});
  mcim2.set({5, 2}, {1, 0});
  mcim2.set({5, 3}, {0, 1});
  mcim2.set({5, 3}, {1, 0});

  MCIM result = mcim1 + mcim2;

  EXPECT_TRUE(result.get({4, 2}, {0, 0}));
  EXPECT_TRUE(result.get({4, 2}, {0, 1}));
  EXPECT_TRUE(result.get({4, 2}, {1, 0}));
  EXPECT_TRUE(result.get({4, 2}, {1, 1}));
  EXPECT_TRUE(result.get({4, 3}, {0, 0}));
  EXPECT_FALSE(result.get({4, 3}, {0, 1}));
  EXPECT_TRUE(result.get({4, 3}, {1, 0}));
  EXPECT_TRUE(result.get({4, 3}, {1, 1}));
  EXPECT_FALSE(result.get({5, 2}, {0, 0}));
  EXPECT_TRUE(result.get({5, 2}, {0, 1}));
  EXPECT_TRUE(result.get({5, 2}, {1, 0}));
  EXPECT_TRUE(result.get({5, 2}, {1, 1}));
  EXPECT_TRUE(result.get({5, 3}, {0, 0}));
  EXPECT_TRUE(result.get({5, 3}, {0, 1}));
  EXPECT_TRUE(result.get({5, 3}, {1, 0}));
  EXPECT_TRUE(result.get({5, 3}, {1, 1}));
}

// Difference of MCIMs.
//
// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      1      0
// (4,3)   1      0      1      0
// (5,2)   0      1      0      1
// (5,3)   1      0      0      1
//
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   1      0      0      1
// (4,3)   1      0      0      1
// (5,2)   0      1      1      0
// (5,3)   0      1      1      0
//
// Expected result:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      1      0
// (4,3)   0      0      1      0
// (5,2)   0      0      0      1
// (5,3)   1      0      0      1

TEST(MCIM_SameRank, difference)
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
  mcim1.set({4, 2}, {0, 1});
  mcim1.set({4, 2}, {1, 0});
  mcim1.set({4, 3}, {0, 0});
  mcim1.set({4, 3}, {1, 0});
  mcim1.set({5, 2}, {0, 1});
  mcim1.set({5, 2}, {1, 1});
  mcim1.set({5, 3}, {0, 0});
  mcim1.set({5, 3}, {1, 1});

  MCIM mcim2(eq, var);
  mcim2.set({4, 2}, {0, 0});
  mcim2.set({4, 2}, {1, 1});
  mcim2.set({4, 3}, {0, 0});
  mcim2.set({4, 3}, {1, 1});
  mcim2.set({5, 2}, {0, 1});
  mcim2.set({5, 2}, {1, 0});
  mcim2.set({5, 3}, {0, 1});
  mcim2.set({5, 3}, {1, 0});

  MCIM result = mcim1 - mcim2;

  EXPECT_FALSE(result.get({4, 2}, {0, 0}));
  EXPECT_TRUE(result.get({4, 2}, {0, 1}));
  EXPECT_TRUE(result.get({4, 2}, {1, 0}));
  EXPECT_FALSE(result.get({4, 2}, {1, 1}));
  EXPECT_FALSE(result.get({4, 3}, {0, 0}));
  EXPECT_FALSE(result.get({4, 3}, {0, 1}));
  EXPECT_TRUE(result.get({4, 3}, {1, 0}));
  EXPECT_FALSE(result.get({4, 3}, {1, 1}));
  EXPECT_FALSE(result.get({5, 2}, {0, 0}));
  EXPECT_FALSE(result.get({5, 2}, {0, 1}));
  EXPECT_FALSE(result.get({5, 2}, {1, 0}));
  EXPECT_TRUE(result.get({5, 2}, {1, 1}));
  EXPECT_TRUE(result.get({5, 3}, {0, 0}));
  EXPECT_FALSE(result.get({5, 3}, {0, 1}));
  EXPECT_FALSE(result.get({5, 3}, {1, 0}));
  EXPECT_TRUE(result.get({5, 3}, {1, 1}));
}

// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)  (4,0)  (4,1)
// (4,2)   0      1      0      0      0      0      1      1      1      1
// (4,3)   0      0      1      0      0      0      1      0      0      1
// (5,2)   0      0      0      1      0      0      0      1      0      1
// (5,3)   0      0      0      0      1      0      0      0      1      1
//
// Expected result:
// {(0,1), (1,0), (1,1), (2,0), (3,0), (3,1), (4,0), (4,1)}

TEST(MCIM_SameRank, flattenRows)
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
  mcim.set({4, 2}, {0, 1});
  mcim.set({4, 2}, {3, 0});
  mcim.set({4, 2}, {3, 1});
  mcim.set({4, 2}, {4, 0});
  mcim.set({4, 2}, {4, 1});
  mcim.set({4, 3}, {1, 0});
  mcim.set({4, 3}, {3, 0});
  mcim.set({4, 3}, {4, 1});
  mcim.set({5, 2}, {1, 1});
  mcim.set({5, 2}, {3, 1});
  mcim.set({5, 2}, {4, 1});
  mcim.set({5, 3}, {2, 0});
  mcim.set({5, 3}, {4, 0});
  mcim.set({5, 3}, {4, 1});

  IndexSet flattened = mcim.flattenRows();

  EXPECT_FALSE(flattened.contains({0, 0}));
  EXPECT_TRUE(flattened.contains({0, 1}));
  EXPECT_TRUE(flattened.contains({1, 0}));
  EXPECT_TRUE(flattened.contains({1, 1}));
  EXPECT_TRUE(flattened.contains({2, 0}));
  EXPECT_FALSE(flattened.contains({2, 1}));
  EXPECT_TRUE(flattened.contains({3, 0}));
  EXPECT_TRUE(flattened.contains({3, 1}));
  EXPECT_TRUE(flattened.contains({4, 0}));
  EXPECT_TRUE(flattened.contains({4, 1}));
}

// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,1)   0      0      0      0
// (4,2)   1      0      0      0
// (4,3)   0      1      0      0
// (5,1)   0      0      1      0
// (5,2)   0      0      0      1
// (5,3)   1      1      0      0
// (6,1)   1      0      1      0
// (6,2)   1      0      0      1
// (6,3)   1      1      1      1
//
// Expected result:
// {(4,1), (4,2), (4,3), (5,1), (5,2), (5,3), (6,1), (6,2), (6,3)}

TEST(MCIM_SameRank, flattenColumns)
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
  mcim.set({4, 2}, {0, 0});
  mcim.set({4, 3}, {0, 1});
  mcim.set({5, 1}, {1, 0});
  mcim.set({5, 2}, {1, 1});
  mcim.set({5, 3}, {0, 0});
  mcim.set({5, 3}, {0, 1});
  mcim.set({6, 1}, {0, 0});
  mcim.set({6, 1}, {1, 0});
  mcim.set({6, 2}, {0, 0});
  mcim.set({6, 2}, {1, 1});
  mcim.set({6, 3}, {0, 0});
  mcim.set({6, 3}, {0, 1});
  mcim.set({6, 3}, {1, 0});
  mcim.set({6, 3}, {1, 1});

  IndexSet flattened = mcim.flattenColumns();

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

// Filter a MCIM by row.
//
// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      0      1
// (4,3)   1      0      1      0
// (5,2)   0      1      0      1
// (5,3)   1      0      1      0
//
// {(4,2), (5,3)}
//
// Expected result:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      0      1
// (4,3)   0      0      0      0
// (5,2)   0      0      0      0
// (5,3)   1      0      1      0

TEST(MCIM_SameRank, rowsFilter)
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
  mcim.set({4, 2}, {0, 1});
  mcim.set({4, 2}, {1, 1});
  mcim.set({4, 3}, {0, 0});
  mcim.set({4, 3}, {1, 0});
  mcim.set({5, 2}, {0, 1});
  mcim.set({5, 2}, {1, 1});
  mcim.set({5, 3}, {0, 0});
  mcim.set({5, 3}, {1, 0});

  IndexSet filter;
  filter += {4, 2};
  filter += {5, 3};

  MCIM result = mcim.filterRows(filter);

  EXPECT_FALSE(result.get({4, 2}, {0, 0}));
  EXPECT_TRUE(result.get({4, 2}, {0, 1}));
  EXPECT_FALSE(result.get({4, 2}, {1, 0}));
  EXPECT_TRUE(result.get({4, 2}, {1, 1}));

  EXPECT_FALSE(result.get({4, 3}, {0, 0}));
  EXPECT_FALSE(result.get({4, 3}, {0, 1}));
  EXPECT_FALSE(result.get({4, 3}, {1, 0}));
  EXPECT_FALSE(result.get({4, 3}, {1, 1}));

  EXPECT_FALSE(result.get({5, 2}, {0, 0}));
  EXPECT_FALSE(result.get({5, 2}, {0, 1}));
  EXPECT_FALSE(result.get({5, 2}, {1, 0}));
  EXPECT_FALSE(result.get({5, 2}, {1, 1}));

  EXPECT_TRUE(result.get({5, 3}, {0, 0}));
  EXPECT_FALSE(result.get({5, 3}, {0, 1}));
  EXPECT_TRUE(result.get({5, 3}, {1, 0}));
  EXPECT_FALSE(result.get({5, 3}, {1, 1}));
}

// Filter a MCIM by column.
//
// Input:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      1      0      1
// (4,3)   1      0      1      0
// (5,2)   0      1      0      1
// (5,3)   1      0      1      0
//
// {(0,0), (1,1)}
//
// Expected result:
// 			 (0,0)  (0,1)  (1,0)  (1,1)
// (4,2)   0      0      0      1
// (4,3)   1      0      0      0
// (5,2)   0      0      0      1
// (5,3)   1      0      0      0

TEST(MCIM_SameRank, columnsFilter)
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
  mcim.set({4, 2}, {0, 1});
  mcim.set({4, 2}, {1, 1});
  mcim.set({4, 3}, {0, 0});
  mcim.set({4, 3}, {1, 0});
  mcim.set({5, 2}, {0, 1});
  mcim.set({5, 2}, {1, 1});
  mcim.set({5, 3}, {0, 0});
  mcim.set({5, 3}, {1, 0});

  IndexSet filter;
  filter += {0, 0};
  filter += {1, 1};

  MCIM result = mcim.filterColumns(filter);

  EXPECT_FALSE(result.get({4, 2}, {0, 0}));
  EXPECT_FALSE(result.get({4, 2}, {0, 1}));
  EXPECT_FALSE(result.get({4, 2}, {1, 0}));
  EXPECT_TRUE(result.get({4, 2}, {1, 1}));

  EXPECT_TRUE(result.get({4, 3}, {0, 0}));
  EXPECT_FALSE(result.get({4, 3}, {0, 1}));
  EXPECT_FALSE(result.get({4, 3}, {1, 0}));
  EXPECT_FALSE(result.get({4, 3}, {1, 1}));

  EXPECT_FALSE(result.get({5, 2}, {0, 0}));
  EXPECT_FALSE(result.get({5, 2}, {0, 1}));
  EXPECT_FALSE(result.get({5, 2}, {1, 0}));
  EXPECT_TRUE(result.get({5, 2}, {1, 1}));

  EXPECT_TRUE(result.get({5, 3}, {0, 0}));
  EXPECT_FALSE(result.get({5, 3}, {0, 1}));
  EXPECT_FALSE(result.get({5, 3}, {1, 0}));
  EXPECT_FALSE(result.get({5, 3}, {1, 1}));
}

//===----------------------------------------------------------------------===//
// Underdimensioned variables
//===----------------------------------------------------------------------===//

TEST(MCIM_UnderdimensionedVariables, indexesIterator)
{
  MultidimensionalRange eq({
      Range(4, 6),
      Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 6));

  MCIM mcim(eq, var);

  std::vector<std::pair<Point, Point>> actual;

  for (auto [equation, variable] : llvm::make_range(mcim.indicesBegin(), mcim.indicesEnd())) {
    actual.emplace_back(equation, variable);
  }

  std::vector<std::pair<Point, Point>> expected;

  for (auto equation : eq) {
    for (auto variable : var) {
      expected.push_back(std::make_pair(equation, variable));
    }
  }

  ASSERT_THAT(actual, testing::UnorderedElementsAreArray(expected.begin(), expected.end()));
}

// Try setting to true the 4 vertices of the matrix.
//
// 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
// (4,2)  1    0    0    0    0    0    0    0    1
// (4,3)  0    0    0    0    0    0    0    0    0
// (5,2)  0    0    0    0    0    0    0    0    0
// (5,3)  1    0    0    0    0    0    0    0    1

TEST(MCIM_UnderdimensionedVariables, set)
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

  EXPECT_TRUE(mcim.get({4, 2}, 0));
  EXPECT_FALSE(mcim.get({4, 2}, 1));
  EXPECT_FALSE(mcim.get({4, 2}, 2));
  EXPECT_FALSE(mcim.get({4, 2}, 3));
  EXPECT_FALSE(mcim.get({4, 2}, 4));
  EXPECT_FALSE(mcim.get({4, 2}, 5));
  EXPECT_FALSE(mcim.get({4, 2}, 6));
  EXPECT_FALSE(mcim.get({4, 2}, 7));
  EXPECT_TRUE(mcim.get({4, 2}, 8));

  EXPECT_FALSE(mcim.get({4, 3}, 0));
  EXPECT_FALSE(mcim.get({4, 3}, 1));
  EXPECT_FALSE(mcim.get({4, 3}, 2));
  EXPECT_FALSE(mcim.get({4, 3}, 3));
  EXPECT_FALSE(mcim.get({4, 3}, 4));
  EXPECT_FALSE(mcim.get({4, 3}, 5));
  EXPECT_FALSE(mcim.get({4, 3}, 6));
  EXPECT_FALSE(mcim.get({4, 3}, 7));
  EXPECT_FALSE(mcim.get({4, 3}, 8));

  EXPECT_FALSE(mcim.get({5, 2}, 0));
  EXPECT_FALSE(mcim.get({5, 2}, 1));
  EXPECT_FALSE(mcim.get({5, 2}, 2));
  EXPECT_FALSE(mcim.get({5, 2}, 3));
  EXPECT_FALSE(mcim.get({5, 2}, 4));
  EXPECT_FALSE(mcim.get({5, 2}, 5));
  EXPECT_FALSE(mcim.get({5, 2}, 6));
  EXPECT_FALSE(mcim.get({5, 2}, 7));
  EXPECT_FALSE(mcim.get({5, 2}, 8));

  EXPECT_TRUE(mcim.get({5, 3}, 0));
  EXPECT_FALSE(mcim.get({5, 3}, 1));
  EXPECT_FALSE(mcim.get({5, 3}, 2));
  EXPECT_FALSE(mcim.get({5, 3}, 3));
  EXPECT_FALSE(mcim.get({5, 3}, 4));
  EXPECT_FALSE(mcim.get({5, 3}, 5));
  EXPECT_FALSE(mcim.get({5, 3}, 6));
  EXPECT_FALSE(mcim.get({5, 3}, 7));
  EXPECT_TRUE(mcim.get({5, 3}, 8));
}

// Sum of MCIMs.
//
// Input:
// 			(0)  (1)  (2)  (3)
// (4,2)  0    1    1    0
// (4,3)  1    0    1    0
// (5,2)  0    1    0    1
// (5,3)  1    0    0    1
//
//			  (0)  (1)  (2)  (3)
// (4,2)  1    0    0    1
// (4,3)  1    0    0    1
// (5,2)  0    1    1    0
// (5,3)  0    1    1    0
//
// Expected result:
// 			 (0)  (1)  (2)  (3)
// (4,2)  1    1    1    1
// (4,3)  1    0    1    1
// (5,2)  0    1    1    1
// (5,3)  1    1    1    1

TEST(MCIM_UnderdimensionedVariables, sum)
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

// Difference of MCIMs.
//
// Input:
//			  (0)  (1)  (2)  (3)
// (4,2)  0    1    1    0
// (4,3)  1    0    1    0
// (5,2)  0    1    0    1
// (5,3)  1    0    0    1
//
// 			 (0)  (1)  (2)  (3)
// (4,2)  1    0    0    1
// (4,3)  1    0    0    1
// (5,2)  0    1    1    0
// (5,3)  0    1    1    0
//
// Expected result:
// 			 (0)  (1)  (2)  (3)
// (4,2)  0    1    1    0
// (4,3)  0    0    1    0
// (5,2)  0    0    0    1
// (5,3)  1    0    0    1

TEST(MCIM_UnderdimensionedVariables, difference)
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

// Input:
// 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
// (4,2)  0    1    0    0    0    1    1    1    1
// (4,3)  0    0    1    0    0    1    0    0    1
// (5,2)  0    0    0    1    0    0    1    0    1
// (5,3)  0    0    0    0    1    0    0    1    1
//
// Expected result:
// {1, 2, 3, 4, 5, 6, 7, 8}

TEST(MCIM_UnderdimensionedVariables, flattenRows)
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

  IndexSet flattened = mcim.flattenRows();

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

// Input:
// 			 (0)  (1)  (2)  (3)
// (4,1)  0    0    0    0
// (4,2)  1    0    0    0
// (4,3)  0    1    0    0
// (5,1)  0    0    1    0
// (5,2)  0    0    0    1
// (5,3)  1    1    0    0
// (6,1)  1    0    1    0
// (6,2)  1    0    0    1
// (6,3)  1    1    1    1
//
// Expected result:
// {(4,2), (4,3), (5,1), (5,2), (5,3), (6,1), (6,2), (6,3)}

TEST(MCIM_UnderdimensionedVariables, flattenColumns)
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

  IndexSet flattened = mcim.flattenColumns();

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

// Filter a MCIM by row.
//
// Input:
// 			 (0)  (1)  (2)  (3)
// (4,2)  0    1    0    1
// (4,3)  1    0    1    0
// (5,2)  0    1    0    1
// (5,3)  1    0    1    0
//
// {(4,2), (5,3)}
//
// Expected result:
// 			 (0)  (1)  (2)  (3)
// (4,2)  0    1    0    1
// (4,3)  0    0    0    0
// (5,2)  0    0    0    0
// (5,3)  1    0    1    0

TEST(MCIM_UnderdimensionedVariables, rowsFilter)
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

  IndexSet filter;
  filter += {4, 2};
  filter += {5, 3};

  MCIM result = mcim.filterRows(filter);

  EXPECT_FALSE(result.get({4, 2}, 0));
  EXPECT_TRUE(result.get({4, 2}, 1));
  EXPECT_FALSE(result.get({4, 2}, 2));
  EXPECT_TRUE(result.get({4, 2}, 3));

  EXPECT_FALSE(result.get({4, 3}, 0));
  EXPECT_FALSE(result.get({4, 3}, 1));
  EXPECT_FALSE(result.get({4, 3}, 2));
  EXPECT_FALSE(result.get({4, 3}, 3));

  EXPECT_FALSE(result.get({5, 2}, 0));
  EXPECT_FALSE(result.get({5, 2}, 1));
  EXPECT_FALSE(result.get({5, 2}, 2));
  EXPECT_FALSE(result.get({5, 2}, 3));

  EXPECT_TRUE(result.get({5, 3}, 0));
  EXPECT_FALSE(result.get({5, 3}, 1));
  EXPECT_TRUE(result.get({5, 3}, 2));
  EXPECT_FALSE(result.get({5, 3}, 3));
}

// Filter a MCIM by column.
//
// Input:
// 			(0)  (1)  (2)  (3)
// (4,2)  0    1    0    1
// (4,3)  1    0    1    0
// (5,2)  0    1    0    1
// (5,3)  1    0    1    0
//
// {0, 3}
//
// Expected result:
// 			 (0)  (1)  (2)  (3)
// (4,2)  0    0    0    1
// (4,3)  1    0    0    0
// (5,2)  0    0    0    1
// (5,3)  1    0    0    0

TEST(MCIM_UnderdimensionedVariables, columnsFilter)
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

  IndexSet filter;
  filter += 0;
  filter += 3;

  MCIM result = mcim.filterColumns(filter);

  EXPECT_FALSE(result.get({4, 2}, 0));
  EXPECT_FALSE(result.get({4, 2}, 1));
  EXPECT_FALSE(result.get({4, 2}, 2));
  EXPECT_TRUE(result.get({4, 2}, 3));

  EXPECT_TRUE(result.get({4, 3}, 0));
  EXPECT_FALSE(result.get({4, 3}, 1));
  EXPECT_FALSE(result.get({4, 3}, 2));
  EXPECT_FALSE(result.get({4, 3}, 3));

  EXPECT_FALSE(result.get({5, 2}, 0));
  EXPECT_FALSE(result.get({5, 2}, 1));
  EXPECT_FALSE(result.get({5, 2}, 2));
  EXPECT_TRUE(result.get({5, 2}, 3));

  EXPECT_TRUE(result.get({5, 3}, 0));
  EXPECT_FALSE(result.get({5, 3}, 1));
  EXPECT_FALSE(result.get({5, 3}, 2));
  EXPECT_FALSE(result.get({5, 3}, 3));
}

//===----------------------------------------------------------------------===//
// Underdimensioned equations
//===----------------------------------------------------------------------===//

TEST(MCIM_UnderdimensionedEquations, indexesIterator)
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

  std::vector<std::pair<Point, Point>> actual;

  for (auto [equation, variable] : llvm::make_range(mcim.indicesBegin(), mcim.indicesEnd())) {
    actual.emplace_back(equation, variable);
  }

  std::vector<std::pair<Point, Point>> expected;

  for (auto equation : eq) {
    for (auto variable : var) {
      expected.push_back(std::make_pair(equation, variable));
    }
  }

  ASSERT_THAT(actual, testing::UnorderedElementsAreArray(expected.begin(), expected.end()));
}

// Try setting to true the 4 vertices of the matrix.
//
// 		 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)
// (3)   1      0      0      0      0      0      0      1
// (4)   0      0      0      0      0      0      0      0
// (5)   0      0      0      0      0      0      0      0
// (6)   1      0      0      0      0      0      0      1

TEST(MCIM_UnderdimensionedEquations, set)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 4),
      Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set(3, {0, 0});
  mcim.set(3, {3, 1});
  mcim.set(6, {0, 0});
  mcim.set(6, {3, 1});

  EXPECT_TRUE(mcim.get(3, {0, 0}));
  EXPECT_FALSE(mcim.get(3, {0, 1}));
  EXPECT_FALSE(mcim.get(3, {1, 0}));
  EXPECT_FALSE(mcim.get(3, {1, 1}));
  EXPECT_FALSE(mcim.get(3, {2, 0}));
  EXPECT_FALSE(mcim.get(3, {2, 1}));
  EXPECT_FALSE(mcim.get(3, {3, 0}));
  EXPECT_TRUE(mcim.get(3, {3, 1}));

  EXPECT_FALSE(mcim.get(4, {0, 0}));
  EXPECT_FALSE(mcim.get(4, {0, 1}));
  EXPECT_FALSE(mcim.get(4, {1, 0}));
  EXPECT_FALSE(mcim.get(4, {1, 1}));
  EXPECT_FALSE(mcim.get(4, {2, 0}));
  EXPECT_FALSE(mcim.get(4, {2, 1}));
  EXPECT_FALSE(mcim.get(4, {3, 0}));
  EXPECT_FALSE(mcim.get(4, {3, 1}));

  EXPECT_FALSE(mcim.get(5, {0, 0}));
  EXPECT_FALSE(mcim.get(5, {0, 1}));
  EXPECT_FALSE(mcim.get(5, {1, 0}));
  EXPECT_FALSE(mcim.get(5, {1, 1}));
  EXPECT_FALSE(mcim.get(5, {2, 0}));
  EXPECT_FALSE(mcim.get(5, {2, 1}));
  EXPECT_FALSE(mcim.get(5, {3, 0}));
  EXPECT_FALSE(mcim.get(5, {3, 1}));

  EXPECT_TRUE(mcim.get(6, {0, 0}));
  EXPECT_FALSE(mcim.get(6, {0, 1}));
  EXPECT_FALSE(mcim.get(6, {1, 0}));
  EXPECT_FALSE(mcim.get(6, {1, 1}));
  EXPECT_FALSE(mcim.get(6, {2, 0}));
  EXPECT_FALSE(mcim.get(6, {2, 1}));
  EXPECT_FALSE(mcim.get(6, {3, 0}));
  EXPECT_TRUE(mcim.get(6, {3, 1}));
}

// Sum of MCIMs.
//
// Input:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      1      0
// (4)   1      0      1      0
// (5)   0      1      0      1
// (6)   1      0      0      1
//
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   1      0      0      1
// (4)   1      0      0      1
// (5)   0      1      1      0
// (6)   0      1      1      0
//
// Expected result:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   1      1      1      1
// (4)   1      0      1      1
// (5)   0      1      1      1
// (6)   1      1      1      1

TEST(MCIM_UnderdimensionedEquations, sum)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 2)
  });

  MCIM mcim1(eq, var);
  mcim1.set(3, {0, 1});
  mcim1.set(3, {1, 0});
  mcim1.set(4, {0, 0});
  mcim1.set(4, {1, 0});
  mcim1.set(5, {0, 1});
  mcim1.set(5, {1, 1});
  mcim1.set(6, {0, 0});
  mcim1.set(6, {1, 1});

  MCIM mcim2(eq, var);
  mcim2.set(3, {0, 0});
  mcim2.set(3, {1, 1});
  mcim2.set(4, {0, 0});
  mcim2.set(4, {1, 1});
  mcim2.set(5, {0, 1});
  mcim2.set(5, {1, 0});
  mcim2.set(6, {0, 1});
  mcim2.set(6, {1, 0});

  MCIM result = mcim1 + mcim2;

  EXPECT_TRUE(result.get(3, {0, 0}));
  EXPECT_TRUE(result.get(3, {0, 1}));
  EXPECT_TRUE(result.get(3, {1, 0}));
  EXPECT_TRUE(result.get(3, {1, 1}));
  EXPECT_TRUE(result.get(4, {0, 0}));
  EXPECT_FALSE(result.get(4, {0, 1}));
  EXPECT_TRUE(result.get(4, {1, 0}));
  EXPECT_TRUE(result.get(4, {1, 1}));
  EXPECT_FALSE(result.get(5, {0, 0}));
  EXPECT_TRUE(result.get(5, {0, 1}));
  EXPECT_TRUE(result.get(5, {1, 0}));
  EXPECT_TRUE(result.get(5, {1, 1}));
  EXPECT_TRUE(result.get(6, {0, 0}));
  EXPECT_TRUE(result.get(6, {0, 1}));
  EXPECT_TRUE(result.get(6, {1, 0}));
  EXPECT_TRUE(result.get(6, {1, 1}));
}

// Difference of MCIMs.
//
// Input:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      1      0
// (4)   1      0      1      0
// (5)   0      1      0      1
// (6)   1      0      0      1
//
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   1      0      0      1
// (4)   1      0      0      1
// (5)   0      1      1      0
// (6)   0      1      1      0
//
// Expected result:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      1      0
// (4)   0      0      1      0
// (5)   0      0      0      1
// (6)   1      0      0      1

TEST(MCIM_UnderdimensionedEquations, difference)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 2)
  });

  MCIM mcim1(eq, var);
  mcim1.set(3, {0, 1});
  mcim1.set(3, {1, 0});
  mcim1.set(4, {0, 0});
  mcim1.set(4, {1, 0});
  mcim1.set(5, {0, 1});
  mcim1.set(5, {1, 1});
  mcim1.set(6, {0, 0});
  mcim1.set(6, {1, 1});

  MCIM mcim2(eq, var);
  mcim2.set(3, {0, 0});
  mcim2.set(3, {1, 1});
  mcim2.set(4, {0, 0});
  mcim2.set(4, {1, 1});
  mcim2.set(5, {0, 1});
  mcim2.set(5, {1, 0});
  mcim2.set(6, {0, 1});
  mcim2.set(6, {1, 0});

  MCIM result = mcim1 - mcim2;

  EXPECT_FALSE(result.get(3, {0, 0}));
  EXPECT_TRUE(result.get(3, {0, 1}));
  EXPECT_TRUE(result.get(3, {1, 0}));
  EXPECT_FALSE(result.get(3, {1, 1}));
  EXPECT_FALSE(result.get(4, {0, 0}));
  EXPECT_FALSE(result.get(4, {0, 1}));
  EXPECT_TRUE(result.get(4, {1, 0}));
  EXPECT_FALSE(result.get(4, {1, 1}));
  EXPECT_FALSE(result.get(5, {0, 0}));
  EXPECT_FALSE(result.get(5, {0, 1}));
  EXPECT_FALSE(result.get(5, {1, 0}));
  EXPECT_TRUE(result.get(5, {1, 1}));
  EXPECT_TRUE(result.get(6, {0, 0}));
  EXPECT_FALSE(result.get(6, {0, 1}));
  EXPECT_FALSE(result.get(6, {1, 0}));
  EXPECT_TRUE(result.get(6, {1, 1}));
}

// Input:
// 		 (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)  (3,0)  (3,1)  (4,0)  (4,1)
// (3)   0      1      0      0      0      0      1      1      1      1
// (4)   0      0      1      0      0      0      1      0      0      1
// (5)   0      0      0      1      0      0      0      1      0      1
// (6)   0      0      0      0      1      0      0      0      1      1
//
// Expected result:
// {(0,1), (1,0), (1,1), (2,0), (3,0), (3,1), (4,0), (4,1)}

TEST(MCIM_UnderdimensionedEquations, flattenRows)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 5),
      Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set(3, {0, 1});
  mcim.set(3, {3, 0});
  mcim.set(3, {3, 1});
  mcim.set(3, {4, 0});
  mcim.set(3, {4, 1});
  mcim.set(4, {1, 0});
  mcim.set(4, {3, 0});
  mcim.set(4, {4, 1});
  mcim.set(5, {1, 1});
  mcim.set(5, {3, 1});
  mcim.set(5, {4, 1});
  mcim.set(6, {2, 0});
  mcim.set(6, {4, 0});
  mcim.set(6, {4, 1});

  IndexSet flattened = mcim.flattenRows();

  EXPECT_FALSE(flattened.contains({0, 0}));
  EXPECT_TRUE(flattened.contains({0, 1}));
  EXPECT_TRUE(flattened.contains({1, 0}));
  EXPECT_TRUE(flattened.contains({1, 1}));
  EXPECT_TRUE(flattened.contains({2, 0}));
  EXPECT_FALSE(flattened.contains({2, 1}));
  EXPECT_TRUE(flattened.contains({3, 0}));
  EXPECT_TRUE(flattened.contains({3, 1}));
  EXPECT_TRUE(flattened.contains({4, 0}));
  EXPECT_TRUE(flattened.contains({4, 1}));
}

// Input:
// 		  (0,0)  (0,1)  (1,0)  (1,1)
// (3)    0      0      0      0
// (4)    1      0      0      0
// (5)    0      1      0      0
// (6)    0      0      1      0
// (7)    0      0      0      1
// (8)    1      1      0      0
// (9)    1      0      1      0
// (10)   1      0      0      1
// (11)   1      1      1      1
//
// Expected result:
// {4, 5, 6, 7, 8, 9, 10, 11}

TEST(MCIM_UnderdimensionedEquations, flattenColumns)
{
  MultidimensionalRange eq(Range(3, 12));

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set(4, {0, 0});
  mcim.set(5, {0, 1});
  mcim.set(6, {1, 0});
  mcim.set(7, {1, 1});
  mcim.set(8, {0, 0});
  mcim.set(8, {0, 1});
  mcim.set(9, {0, 0});
  mcim.set(9, {1, 0});
  mcim.set(10, {0, 0});
  mcim.set(10, {1, 1});
  mcim.set(11, {0, 0});
  mcim.set(11, {0, 1});
  mcim.set(11, {1, 0});
  mcim.set(11, {1, 1});

  IndexSet flattened = mcim.flattenColumns();

  EXPECT_FALSE(flattened.contains(3));
  EXPECT_TRUE(flattened.contains(4));
  EXPECT_TRUE(flattened.contains(5));
  EXPECT_TRUE(flattened.contains(6));
  EXPECT_TRUE(flattened.contains(7));
  EXPECT_TRUE(flattened.contains(8));
  EXPECT_TRUE(flattened.contains(9));
  EXPECT_TRUE(flattened.contains(10));
  EXPECT_TRUE(flattened.contains(11));
}

// Filter a MCIM by row.
//
// Input:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      0      1
// (4)   1      0      1      0
// (5)   0      1      0      1
// (6)   1      0      1      0
//
// {3, 6}
//
// Expected result:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      0      1
// (4)   0      0      0      0
// (5)   0      0      0      0
// (6)   1      0      1      0

TEST(MCIM_UnderdimensionedEquations, rowsFilter)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set(3, {0, 1});
  mcim.set(3, {1, 1});
  mcim.set(4, {0, 0});
  mcim.set(4, {1, 0});
  mcim.set(5, {0, 1});
  mcim.set(5, {1, 1});
  mcim.set(6, {0, 0});
  mcim.set(6, {1, 0});

  IndexSet filter;
  filter += 3;
  filter += 6;

  MCIM result = mcim.filterRows(filter);

  EXPECT_FALSE(result.get(3, {0, 0}));
  EXPECT_TRUE(result.get(3, {0, 1}));
  EXPECT_FALSE(result.get(3, {1, 0}));
  EXPECT_TRUE(result.get(3, {1, 1}));

  EXPECT_FALSE(result.get(4, {0, 0}));
  EXPECT_FALSE(result.get(4, {0, 1}));
  EXPECT_FALSE(result.get(4, {1, 0}));
  EXPECT_FALSE(result.get(4, {1, 1}));

  EXPECT_FALSE(result.get(5, {0, 0}));
  EXPECT_FALSE(result.get(5, {0, 1}));
  EXPECT_FALSE(result.get(5, {1, 0}));
  EXPECT_FALSE(result.get(5, {1, 1}));

  EXPECT_TRUE(result.get(6, {0, 0}));
  EXPECT_FALSE(result.get(6, {0, 1}));
  EXPECT_TRUE(result.get(6, {1, 0}));
  EXPECT_FALSE(result.get(6, {1, 1}));
}

// Filter a MCIM by column.
//
// Input:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      1      0      1
// (4)   1      0      1      0
// (5)   0      1      0      1
// (6)   1      0      1      0
//
// {(0,0), (1,1)}
//
// Expected result:
// 		 (0,0)  (0,1)  (1,0)  (1,1)
// (3)   0      0      0      1
// (4)   1      0      0      0
// (5)   0      0      0      1
// (6)   1      0      0      0

TEST(MCIM_UnderdimensionedEquations, columnsFilter)
{
  MultidimensionalRange eq(Range(3, 7));

  MultidimensionalRange var({
      Range(0, 2),
      Range(0, 2)
  });

  MCIM mcim(eq, var);
  mcim.set(3, {0, 1});
  mcim.set(3, {1, 1});
  mcim.set(4, {0, 0});
  mcim.set(4, {1, 0});
  mcim.set(5, {0, 1});
  mcim.set(5, {1, 1});
  mcim.set(6, {0, 0});
  mcim.set(6, {1, 0});

  IndexSet filter;
  filter += {0, 0};
  filter += {1, 1};

  MCIM result = mcim.filterColumns(filter);

  EXPECT_FALSE(result.get(3, {0, 0}));
  EXPECT_FALSE(result.get(3, {0, 1}));
  EXPECT_FALSE(result.get(3, {1, 0}));
  EXPECT_TRUE(result.get(3, {1, 1}));

  EXPECT_TRUE(result.get(4, {0, 0}));
  EXPECT_FALSE(result.get(4, {0, 1}));
  EXPECT_FALSE(result.get(4, {1, 0}));
  EXPECT_FALSE(result.get(4, {1, 1}));

  EXPECT_FALSE(result.get(5, {0, 0}));
  EXPECT_FALSE(result.get(5, {0, 1}));
  EXPECT_FALSE(result.get(5, {1, 0}));
  EXPECT_TRUE(result.get(5, {1, 1}));

  EXPECT_TRUE(result.get(6, {0, 0}));
  EXPECT_FALSE(result.get(6, {0, 1}));
  EXPECT_FALSE(result.get(6, {1, 0}));
  EXPECT_FALSE(result.get(6, {1, 1}));
}
