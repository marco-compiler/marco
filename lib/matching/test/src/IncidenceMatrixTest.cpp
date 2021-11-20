#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/IncidenceMatrix.h>
#include <vector>

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, incidenceMatrixIndexesIterator)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 3)
  });

  IncidenceMatrix matrix(eq, var);

  llvm::SmallVector<std::vector<long>, 8> expectedList;

  for (auto equationIndexes : eq)
  {
    for (auto variableIndexes : var)
    {
      std::vector<long> current;

      for (const auto& index : equationIndexes)
        current.push_back(index);

      for (const auto& index : variableIndexes)
        current.push_back(index);

      expectedList.push_back(std::move(current));
    }
  }

  size_t counter = 0;

  for (auto indexes : matrix.getIndexes())
  {
    ASSERT_LT(counter, expectedList.size());
    auto& expected = expectedList[counter];
    EXPECT_EQ(indexes.size(), expected.size());
    EXPECT_THAT(indexes, testing::ContainerEq(expected));
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
TEST(Matching, incidenceMatrixEdgesSet)
{
	MultidimensionalRange eq({
			Range(4, 6),
			Range(2, 4)
	});

	MultidimensionalRange var(Range(0, 9));

	IncidenceMatrix matrix(eq, var);
	matrix.set({ 4, 2, 0 });
	matrix.set({ 4, 2, 8 });
	matrix.set({ 5, 3, 0 });
	matrix.set({ 5, 3, 8 });

	for (long i = 4; i < 6; ++i)
	{
		for (long j = 2; j < 4; ++j)
		{
			for (long k = 0; k < 9; ++k)
			{
				bool value = matrix.get({ i, j, k });

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
 * 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
 * (4,2)  0    1    0    0    0    1    1    1    1
 * (4,3)  0    0    1    0    0    1    0    0    1
 * (5,2)  0    0    0    1    0    0    1    0    1
 * (5,3)  0    0    0    0    1    0    0    1    1
 *
 * Expected result:
 * 			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)  (8)
 * (0)    0    1    1    1    1    1    1    1    1
 */
TEST(Matching, incidenceMatrixFlattenEquations)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 9));

  IncidenceMatrix matrix(eq, var);
  matrix.set({ 4, 2, 1 });
  matrix.set({ 4, 3, 2 });
  matrix.set({ 5, 2, 3 });
  matrix.set({ 5, 3, 4 });
  matrix.set({ 4, 2, 5 });
  matrix.set({ 4, 3, 5 });
  matrix.set({ 4, 2, 6 });
  matrix.set({ 5, 2, 6 });
  matrix.set({ 4, 2, 7 });
  matrix.set({ 5, 3, 7 });
  matrix.set({ 4, 2, 8 });
  matrix.set({ 4, 3, 8 });
  matrix.set({ 5, 2, 8 });
  matrix.set({ 5, 3, 8 });

  IncidenceMatrix flattened = matrix.flattenEquations();
  EXPECT_FALSE(flattened.get({ 0, 0 }));
  EXPECT_TRUE(flattened.get({ 0, 1 }));
  EXPECT_TRUE(flattened.get({ 0, 2 }));
  EXPECT_TRUE(flattened.get({ 0, 3 }));
  EXPECT_TRUE(flattened.get({ 0, 4 }));
  EXPECT_TRUE(flattened.get({ 0, 5 }));
  EXPECT_TRUE(flattened.get({ 0, 6 }));
  EXPECT_TRUE(flattened.get({ 0, 7 }));
  EXPECT_TRUE(flattened.get({ 0, 8 }));
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
 * 			 (0)
 * (4,1)  0
 * (4,2)  1
 * (4,3)  1
 * (5,1)  1
 * (5,2)  1
 * (5,3)  1
 * (6,1)  1
 * (6,2)  1
 * (6,3)  1
 */
TEST(Matching, incidenceMatrixFlattenVariables)
{
  MultidimensionalRange eq({
    Range(4, 7),
    Range(1, 4)
  });

  MultidimensionalRange var(Range(0, 4));

  IncidenceMatrix matrix(eq, var);
  matrix.set({ 4, 2, 0 });
  matrix.set({ 4, 3, 1 });
  matrix.set({ 5, 1, 2 });
  matrix.set({ 5, 2, 3 });
  matrix.set({ 5, 3, 0 });
  matrix.set({ 5, 3, 1 });
  matrix.set({ 6, 1, 0 });
  matrix.set({ 6, 1, 2 });
  matrix.set({ 6, 2, 0 });
  matrix.set({ 6, 2, 3 });
  matrix.set({ 6, 3, 0 });
  matrix.set({ 6, 3, 1 });
  matrix.set({ 6, 3, 2 });
  matrix.set({ 6, 3, 3 });

  IncidenceMatrix flattened = matrix.flattenVariables();
  EXPECT_FALSE(flattened.get({ 4, 1, 0 }));
  EXPECT_TRUE(flattened.get({ 4, 2, 0 }));
  EXPECT_TRUE(flattened.get({ 4, 3, 0 }));
  EXPECT_TRUE(flattened.get({ 5, 1, 0 }));
  EXPECT_TRUE(flattened.get({ 5, 2, 0 }));
  EXPECT_TRUE(flattened.get({ 5, 3, 0 }));
  EXPECT_TRUE(flattened.get({ 6, 1, 0 }));
  EXPECT_TRUE(flattened.get({ 6, 2, 0 }));
  EXPECT_TRUE(flattened.get({ 6, 3, 0 }));
}

/**
 * Sum of incidence matrices.
 *
 * Input:
 * 			 (0)  (1)
 * (4,2)  0    1
 * (4,3)  1    0
 * (5,2)  0    1
 * (5,3)  1    0
 *
 * 			 (0)  (1)
 * (4,2)  1    0
 * (4,3)  1    0
 * (5,2)  0    1
 * (5,3)  0    1
 *
 * Expected result:
 * 			 (0)  (1)
 * (4,2)  1    1
 * (4,3)  1    0
 * (5,2)  0    1
 * (5,3)  1    1
 */
TEST(Matching, incidenceMatrixSum)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 2));

  IncidenceMatrix matrix1(eq, var);
  matrix1.set({ 4, 2, 1 });
  matrix1.set({ 4, 3, 0 });
  matrix1.set({ 5, 2, 1 });
  matrix1.set({ 5, 3, 0 });

  IncidenceMatrix matrix2(eq, var);
  matrix2.set({ 4, 2, 0 });
  matrix2.set({ 4, 3, 0 });
  matrix2.set({ 5, 2, 1 });
  matrix2.set({ 5, 3, 1 });

  IncidenceMatrix result = matrix1 + matrix2;
  EXPECT_TRUE(result.get({ 4, 2, 0 }));
  EXPECT_TRUE(result.get({ 4, 2, 1 }));
  EXPECT_TRUE(result.get({ 4, 3, 0 }));
  EXPECT_FALSE(result.get({ 4, 3, 1 }));
  EXPECT_FALSE(result.get({ 5, 2, 0 }));
  EXPECT_TRUE(result.get({ 5, 2, 1 }));
  EXPECT_TRUE(result.get({ 5, 3, 0 }));
  EXPECT_TRUE(result.get({ 5, 3, 1 }));
}

/**
 * Difference of incidence matrices.
 *
 * Input:
 * 			 (0)  (1)
 * (4,2)  0    1
 * (4,3)  1    0
 * (5,2)  0    1
 * (5,3)  1    0
 *
 * 			 (0)  (1)
 * (4,2)  1    0
 * (4,3)  1    0
 * (5,2)  0    1
 * (5,3)  0    1
 *
 * Expected result:
 * 			 (0)  (1)
 * (4,2)  0    1
 * (4,3)  0    0
 * (5,2)  0    0
 * (5,3)  1    0
 */
TEST(Matching, incidenceMatrixDifference)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var(Range(0, 2));

  IncidenceMatrix matrix1(eq, var);
  matrix1.set({ 4, 2, 1 });
  matrix1.set({ 4, 3, 0 });
  matrix1.set({ 5, 2, 1 });
  matrix1.set({ 5, 3, 0 });

  IncidenceMatrix matrix2(eq, var);
  matrix2.set({ 4, 2, 0 });
  matrix2.set({ 4, 3, 0 });
  matrix2.set({ 5, 2, 1 });
  matrix2.set({ 5, 3, 1 });

  IncidenceMatrix result = matrix1 - matrix2;
  EXPECT_FALSE(result.get({ 4, 2, 0 }));
  EXPECT_TRUE(result.get({ 4, 2, 1 }));
  EXPECT_FALSE(result.get({ 4, 3, 0 }));
  EXPECT_FALSE(result.get({ 4, 3, 1 }));
  EXPECT_FALSE(result.get({ 5, 2, 0 }));
  EXPECT_FALSE(result.get({ 5, 2, 1 }));
  EXPECT_TRUE(result.get({ 5, 3, 0 }));
  EXPECT_FALSE(result.get({ 5, 3, 1 }));
}

/**
 * Filter an incidence matrix by equation.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      1      0
 *
 * 			 (0)
 * (4,2)  1
 * (4,3)  0
 * (5,2)  0
 * (5,3)  1
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   0      0      0      0
 * (5,2)   0      0      0      0
 * (5,3)   1      0      1      0
 */
TEST(Matching, incidenceMatrixEquationFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  IncidenceMatrix matrix(eq, var);
  matrix.set({ 4, 2, 0, 1 });
  matrix.set({ 4, 2, 1, 1 });
  matrix.set({ 4, 3, 0, 0 });
  matrix.set({ 4, 3, 1, 0 });
  matrix.set({ 5, 2, 0, 1 });
  matrix.set({ 5, 2, 1, 1 });
  matrix.set({ 5, 3, 0, 0 });
  matrix.set({ 5, 3, 1, 0 });

  MultidimensionalRange filterEq({
    Range(4, 6),
    Range(2, 4)
  });

  IncidenceMatrix filter = IncidenceMatrix::column(eq);
  filter.set({ 4, 2, 0 });
  filter.set({ 5, 3, 0 });

  IncidenceMatrix result = matrix.filterEquations(filter);

  for (const auto& indexes : result.getIndexes())
  {
    if (indexes[0] == 4 && indexes[1] == 2 && indexes[2] == 0 && indexes[3] == 1)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 4 && indexes[1] == 2 && indexes[2] == 1 && indexes[3] == 1)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 5 && indexes[1] == 3 && indexes[2] == 0 && indexes[3] == 0)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 5 && indexes[1] == 3 && indexes[2] == 1 && indexes[3] == 0)
      EXPECT_TRUE(result.get(indexes));
    else
      EXPECT_FALSE(result.get(indexes));
  }
}

/**
 * Filter an incidence matrix by variable.
 *
 * Input:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      1      0      1
 * (4,3)   1      0      1      0
 * (5,2)   0      1      0      1
 * (5,3)   1      0      1      0
 *
 * 		 (0,0)  (0,1)  (1,0)  (1,1)
 * (0)   1      0      0      1
 *
 * Expected result:
 * 			 (0,0)  (0,1)  (1,0)  (1,1)
 * (4,2)   0      0      0      1
 * (4,3)   1      0      0      0
 * (5,2)   0      0      0      1
 * (5,3)   1      0      0      0
 */
TEST(Matching, incidenceMatrixVariableFilter)
{
  MultidimensionalRange eq({
    Range(4, 6),
    Range(2, 4)
  });

  MultidimensionalRange var({
    Range(0, 2),
    Range(0, 2)
  });

  IncidenceMatrix matrix(eq, var);
  matrix.set({ 4, 2, 0, 1 });
  matrix.set({ 4, 2, 1, 1 });
  matrix.set({ 4, 3, 0, 0 });
  matrix.set({ 4, 3, 1, 0 });
  matrix.set({ 5, 2, 0, 1 });
  matrix.set({ 5, 2, 1, 1 });
  matrix.set({ 5, 3, 0, 0 });
  matrix.set({ 5, 3, 1, 0 });

  MultidimensionalRange filterEq({
    Range(4, 6),
    Range(2, 4)
  });

  IncidenceMatrix filter = IncidenceMatrix::row(var);
  filter.set({ 0, 0, 0 });
  filter.set({ 0, 1, 1 });

  IncidenceMatrix result = matrix.filterVariables(filter);

  for (const auto& indexes : result.getIndexes())
  {
    if (indexes[0] == 4 && indexes[1] == 2 && indexes[2] == 1 && indexes[3] == 1)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 4 && indexes[1] == 3 && indexes[2] == 0 && indexes[3] == 0)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 5 && indexes[1] == 2 && indexes[2] == 1 && indexes[3] == 1)
      EXPECT_TRUE(result.get(indexes));
    else if (indexes[0] == 5 && indexes[1] == 3 && indexes[2] == 0 && indexes[3] == 0)
      EXPECT_TRUE(result.get(indexes));
    else
      EXPECT_FALSE(result.get(indexes));
  }
}
