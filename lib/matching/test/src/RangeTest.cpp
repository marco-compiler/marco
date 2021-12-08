#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/Range.h>

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

TEST(Matching, oneDimensionalRangeContainsValue)
{
  Range range(3, 6);

  EXPECT_FALSE(range.contains(2));
  EXPECT_TRUE(range.contains(3));
  EXPECT_TRUE(range.contains(4));
  EXPECT_TRUE(range.contains(5));
  EXPECT_FALSE(range.contains(6));
}

TEST(Matching, oneDimensionalRangeContainsRange)
{
  Range range(3, 6);

  EXPECT_FALSE(range.contains(Range(1, 2)));
  EXPECT_FALSE(range.contains(Range(1, 3)));
  EXPECT_FALSE(range.contains(Range(1, 4)));
  EXPECT_TRUE(range.contains(Range(3, 5)));
  EXPECT_TRUE(range.contains(Range(4, 6)));
  EXPECT_TRUE(range.contains(Range(3, 6)));
  EXPECT_FALSE(range.contains(Range(5, 7)));
  EXPECT_FALSE(range.contains(Range(6, 9)));
  EXPECT_FALSE(range.contains(Range(7, 9)));
}

TEST(Matching, oneDimensionalOverlappingRanges)
{
  Range x(1, 5);
  Range y(2, 7);

  EXPECT_TRUE(x.overlaps(y));
  EXPECT_TRUE(y.overlaps(x));
}

TEST(Matching, oneDimensionalRangesWithTouchingBordersDoNotOverlap)
{
  Range x(1, 5);
  Range y(5, 7);

  EXPECT_FALSE(x.overlaps(y));
  EXPECT_FALSE(y.overlaps(x));
}

TEST(Matching, oneDimensionalRangesMerging)
{
  Range x(1, 5);

  // Overlapping
  Range y(3, 11);

  EXPECT_TRUE(x.canBeMerged(y));
  EXPECT_TRUE(y.canBeMerged(x));

  EXPECT_EQ(x.merge(y).getBegin(), 1);
  EXPECT_EQ(x.merge(y).getEnd(), 11);

  EXPECT_EQ(y.merge(x).getBegin(), 1);
  EXPECT_EQ(y.merge(x).getEnd(), 11);

  // Touching borders
  Range z(5, 7);

  EXPECT_TRUE(x.canBeMerged(z));
  EXPECT_TRUE(z.canBeMerged(x));

  EXPECT_EQ(x.merge(z).getBegin(), 1);
  EXPECT_EQ(x.merge(z).getEnd(), 7);

  EXPECT_EQ(z.merge(x).getBegin(), 1);
  EXPECT_EQ(z.merge(x).getEnd(), 7);
}

TEST(Matching, oneDimensionalRangesSubtraction)
{
  Range a(3, 7);

  // Overlapping
  Range b(5, 11);

  EXPECT_THAT(a.subtract(b), testing::UnorderedElementsAre(Range(3, 5)));
  EXPECT_THAT(b.subtract(a), testing::UnorderedElementsAre(Range(7, 11)));

  // Fully contained
  Range c(2, 11);
  Range d(2, 5);
  Range e(7, 11);

  EXPECT_THAT(a.subtract(c), testing::IsEmpty());
  EXPECT_THAT(c.subtract(a), testing::UnorderedElementsAre(Range(2, 3), Range(7, 11)));
  EXPECT_THAT(c.subtract(d), testing::UnorderedElementsAre(Range(5, 11)));
  EXPECT_THAT(c.subtract(e), testing::UnorderedElementsAre(Range(2, 7)));
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

TEST(Matching, multidimensionalRangesOverlap)
{
  MultidimensionalRange x({
    Range(1, 3),
    Range(2, 4)
  });

  MultidimensionalRange y({
    Range(2, 3),
    Range(3, 5)
  });

  EXPECT_TRUE(x.overlaps(y));
  EXPECT_TRUE(y.overlaps(x));
}

TEST(Matching, multidimensionalRangesWithTouchingBordersDoNotOverlap)
{
  MultidimensionalRange x({
    Range(1, 3),
    Range(2, 4)
  });

  MultidimensionalRange y({
    Range(3, 5),
    Range(3, 5)
  });

  EXPECT_FALSE(x.overlaps(y));
  EXPECT_FALSE(y.overlaps(x));
}

TEST(Matching, multidimensionalRangesWithTouchingBordersMerging)
{
  MultidimensionalRange x({
    Range(1, 3),
    Range(2, 4)
  });

  MultidimensionalRange y({
    Range(1, 3),
    Range(4, 7)
  });

  EXPECT_TRUE(x.canBeMerged(y).first);
  EXPECT_TRUE(y.canBeMerged(x).first);

  MultidimensionalRange z = x.merge(y, x.canBeMerged(y).second);

  EXPECT_EQ(z[0].getBegin(), 1);
  EXPECT_EQ(z[0].getEnd(), 3);
  EXPECT_EQ(z[1].getBegin(), 2);
  EXPECT_EQ(z[1].getEnd(), 7);

  MultidimensionalRange t = x.merge(y, x.canBeMerged(y).second);
  EXPECT_EQ(z, t);
}

TEST(Matching, multidimensionalRangesWithOverlapMerging)
{
  MultidimensionalRange x({
    Range(1, 3),
    Range(2, 6)
  });

  MultidimensionalRange y({
    Range(1, 3),
    Range(4, 7)
  });

  EXPECT_TRUE(x.canBeMerged(y).first);
  EXPECT_TRUE(y.canBeMerged(x).first);

  MultidimensionalRange z = x.merge(y, x.canBeMerged(y).second);

  EXPECT_EQ(z[0].getBegin(), 1);
  EXPECT_EQ(z[0].getEnd(), 3);
  EXPECT_EQ(z[1].getBegin(), 2);
  EXPECT_EQ(z[1].getEnd(), 7);

  MultidimensionalRange t = x.merge(y, x.canBeMerged(y).second);
  EXPECT_EQ(z, t);
}

TEST(Matching, multidimensionalRangesUnmergeable)
{
  MultidimensionalRange x({
    Range(1, 3),
    Range(2, 4)
  });

  MultidimensionalRange y({
    Range(3, 5),
    Range(4, 7)
  });

  EXPECT_FALSE(x.canBeMerged(y).first);
  EXPECT_FALSE(y.canBeMerged(x).first);
}

TEST(Matching, multiDimensionalRangesSubtraction)
{
  MultidimensionalRange a({
    Range(2, 10),
    Range(3, 7),
    Range(1, 8)
  });

  // Fully contained in 'a'
  MultidimensionalRange b({
    Range(6, 8),
    Range(4, 6),
    Range(3, 7)
  });

  EXPECT_THAT(a.subtract(b), testing::UnorderedElementsAre(
          MultidimensionalRange({
            Range(2, 6),
            Range(3, 7),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(8, 10),
            Range(3, 7),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(3, 4),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(6, 7),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(4, 6),
            Range(1, 3)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(4, 6),
            Range(7, 8)
          })));

  // 2 dimensions fully contained, 1 fully traversing
  MultidimensionalRange c({
    Range(6, 8),
    Range(1, 9),
    Range(3, 7)
  });

  EXPECT_THAT(a.subtract(c), testing::UnorderedElementsAre(
          MultidimensionalRange({
            Range(2, 6),
            Range(3, 7),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(8, 10),
            Range(3, 7),
            Range(1, 8)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(3, 7),
            Range(1, 3)
          }),
          MultidimensionalRange({
            Range(6, 8),
            Range(3, 7),
            Range(7, 8)
          })));

  // 1 dimension fully contained, 2 fully traversing
  MultidimensionalRange d({
    Range(1, 15),
    Range(1, 9),
    Range(3, 7)
  });

  EXPECT_THAT(a.subtract(d), testing::UnorderedElementsAre(
          MultidimensionalRange({
            Range(2, 10),
            Range(3, 7),
            Range(1, 3)
          }),
          MultidimensionalRange({
            Range(2, 10),
            Range(3, 7),
            Range(7, 8)
          })));

  // 3 dimensions fully traversing
  MultidimensionalRange e({
    Range(1, 15),
    Range(1, 9),
    Range(0, 11)
  });

  EXPECT_THAT(a.subtract(e), testing::IsEmpty());
}
