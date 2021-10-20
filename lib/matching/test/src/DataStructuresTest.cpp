#include <gtest/gtest.h>
#include <marco/matching/Range.h>

using namespace marco::matching;

TEST(Matching, oneDimensionalRange)
{
	Range range(1, 5);

	ASSERT_EQ(range.getBegin(), 1);
	ASSERT_EQ(range.getEnd(), 5);
	ASSERT_EQ(range.size(), 4);
}

TEST(Matching, oneDimensionalIntersectingRanges)
{
	Range x(1, 5);
	Range y(2, 6);

	EXPECT_TRUE(x.intersects(y));
}

TEST(Matching, oneDimensionalRangesWithTouchingBorders)
{
	Range x(1, 5);
	Range y(5, 7);

	EXPECT_FALSE(x.intersects(y));
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
}

