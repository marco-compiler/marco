#include <gtest/gtest.h>
#include <marco/matching/IncidenceMatrix.h>

using namespace marco::matching;

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
