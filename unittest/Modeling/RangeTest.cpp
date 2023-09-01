#include "marco/Modeling/Range.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ::marco::modeling;

TEST(Range, borders)
{
  Range range(1, 5);

  EXPECT_EQ(range.getBegin(), 1);
  EXPECT_EQ(range.getEnd(), 5);
}

TEST(Range, size)
{
  Range range(1, 5);
  EXPECT_EQ(range.size(), 4);
}

TEST(Range, iteration)
{
  Range range(1, 5);

  auto begin = range.begin();
  auto end = range.end();

  auto value = range.getBegin();

  for (auto it = begin; it != end; ++it) {
    EXPECT_EQ(*it, value++);
  }
}

TEST(Range, containsValue)
{
  Range range(3, 6);

  EXPECT_FALSE(range.contains(2));
  EXPECT_TRUE(range.contains(3));
  EXPECT_TRUE(range.contains(4));
  EXPECT_TRUE(range.contains(5));
  EXPECT_FALSE(range.contains(6));
}

TEST(Range, containsRange)
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

TEST(Range, overlap)
{
  Range x(1, 5);
  Range y(2, 7);

  EXPECT_TRUE(x.overlaps(y));
  EXPECT_TRUE(y.overlaps(x));
}

TEST(Range, touchingBordersDoNotOverlap)
{
  Range x(1, 5);
  Range y(5, 7);

  EXPECT_FALSE(x.overlaps(y));
  EXPECT_FALSE(y.overlaps(x));
}

TEST(Range, merge)
{
  Range x(1, 5);

  // Overlapping.
  Range y(3, 11);

  EXPECT_TRUE(x.canBeMerged(y));
  EXPECT_TRUE(y.canBeMerged(x));

  EXPECT_EQ(x.merge(y).getBegin(), 1);
  EXPECT_EQ(x.merge(y).getEnd(), 11);

  EXPECT_EQ(y.merge(x).getBegin(), 1);
  EXPECT_EQ(y.merge(x).getEnd(), 11);

  // Touching borders.
  Range z(5, 7);

  EXPECT_TRUE(x.canBeMerged(z));
  EXPECT_TRUE(z.canBeMerged(x));

  EXPECT_EQ(x.merge(z).getBegin(), 1);
  EXPECT_EQ(x.merge(z).getEnd(), 7);

  EXPECT_EQ(z.merge(x).getBegin(), 1);
  EXPECT_EQ(z.merge(x).getEnd(), 7);
}

TEST(Range, subtraction)
{
  Range a(3, 7);

  // Overlapping.
  Range b(5, 11);

  EXPECT_THAT(a.subtract(b), testing::UnorderedElementsAre(Range(3, 5)));
  EXPECT_THAT(b.subtract(a), testing::UnorderedElementsAre(Range(7, 11)));

  // Fully contained.
  Range c(2, 11);
  Range d(2, 5);
  Range e(7, 11);

  EXPECT_THAT(a.subtract(c), testing::IsEmpty());
  EXPECT_THAT(c.subtract(a), testing::UnorderedElementsAre(Range(2, 3), Range(7, 11)));
  EXPECT_THAT(c.subtract(d), testing::UnorderedElementsAre(Range(5, 11)));
  EXPECT_THAT(c.subtract(e), testing::UnorderedElementsAre(Range(2, 7)));
}
