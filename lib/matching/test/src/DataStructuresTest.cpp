#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/IncidenceMatrix.h>
#include <vector>

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, oneDimensionalRange)
{
	Range range(1, 5);

	ASSERT_EQ(range.getBegin(), 1);
	ASSERT_EQ(range.getEnd(), 5);
	ASSERT_EQ(range.size(), 4);
}

TEST(Matching, oneDimensionalRangeIteration)
{
	Range range(1, 5);

	auto begin = range.begin();
	auto end = range.end();

	Range::data_type value = range.getBegin();

	for (auto it = begin; it != end; ++it)
		EXPECT_EQ(*it, value++);
}

TEST(Matching, oneDimensionalIntersectingRanges)
{
	Range x(1, 5);
	Range y(2, 7);

	EXPECT_TRUE(x.intersects(y));
	EXPECT_TRUE(y.intersects(x));
}

TEST(Matching, oneDimensionalRangesWithTouchingBorders)
{
	Range x(1, 5);
	Range y(5, 7);

	EXPECT_FALSE(x.intersects(y));
	EXPECT_FALSE(y.intersects(x));
}

TEST(Matching, multidimensionalRange)
{
	MultidimensionalRange range({
			Range(1, 3),
			Range(2, 5),
			Range(7, 10)
	});

	EXPECT_EQ(range.rank(), 3);
	EXPECT_EQ(range.flatSize(), 18);
}

TEST(Matching, multiDimensionalRangeIteration)
{
	MultidimensionalRange range({
			Range(1, 3),
			Range(2, 5),
			Range(8, 10)
	});

	llvm::SmallVector<std::tuple<long, long, long>, 3> expected;
	expected.emplace_back(1, 2, 8);
	expected.emplace_back(1, 2, 9);
	expected.emplace_back(1, 3, 8);
	expected.emplace_back(1, 3, 9);
	expected.emplace_back(1, 4, 8);
	expected.emplace_back(1, 4, 9);
	expected.emplace_back(2, 2, 8);
	expected.emplace_back(2, 2, 9);
	expected.emplace_back(2, 3, 8);
	expected.emplace_back(2, 3, 9);
	expected.emplace_back(2, 4, 8);
	expected.emplace_back(2, 4, 9);

	size_t index = 0;

	for (auto it = range.begin(), end = range.end(); it != end; ++it)
	{
		auto values = *it;

		EXPECT_EQ(values.size(), 3);
		EXPECT_EQ(values[0], std::get<0>(expected[index]));
		EXPECT_EQ(values[1], std::get<1>(expected[index]));
		EXPECT_EQ(values[2], std::get<2>(expected[index]));

		++index;
	}
}

TEST(Matching, multidimensionalRangesIntersection)
{
	MultidimensionalRange x({
			Range(1, 3),
			Range(2, 4)
	});

	MultidimensionalRange y({
			Range(2, 3),
			Range(3, 5)
	});

	EXPECT_TRUE(x.intersects(y));
	EXPECT_TRUE(y.intersects(x));
}

TEST(Matching, multidimensionalRangesWithTouchingBorders)
{
	MultidimensionalRange x({
			Range(1, 3),
			Range(2, 4)
	});

	MultidimensionalRange y({
			Range(3, 4),
			Range(3, 5)
	});

	EXPECT_FALSE(x.intersects(y));
	EXPECT_FALSE(y.intersects(x));
}

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
 * Try setting to true the 4 vertices of the flattened matrix.
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
 * Try setting to true the 4 vertices of the flattened matrix.
 *
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
 * Try setting to true the 4 vertices of the flattened matrix.
 *
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
