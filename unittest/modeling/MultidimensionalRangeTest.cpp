#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/modeling/MultidimensionalRange.h"

using namespace ::marco::modeling::internal;

TEST(MultidimensionalRange, rank)
{
  MultidimensionalRange range({
      Range(1, 3),
      Range(2, 5),
      Range(7, 10)
  });

  EXPECT_EQ(range.rank(), 3);
}

TEST(MultidimensionalRange, flatSize)
{
  MultidimensionalRange range({
      Range(1, 3),
      Range(2, 5),
      Range(7, 10)
  });

  EXPECT_EQ(range.flatSize(), 18);
}

TEST(MultidimensionalRange, iteration)
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

  for (auto it = range.begin(), end = range.end(); it != end; ++it) {
    auto values = *it;

    EXPECT_EQ(values.rank(), 3);
    EXPECT_EQ(values[0], std::get<0>(expected[index]));
    EXPECT_EQ(values[1], std::get<1>(expected[index]));
    EXPECT_EQ(values[2], std::get<2>(expected[index]));

    ++index;
  }
}

TEST(MultidimensionalRange, overlap)
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

TEST(MultidimensionalRange, touchingBordersDoNotOverlap)
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

TEST(MultidimensionalRange, canMergeWithTouchingBorders)
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

TEST(MultidimensionalRange, canMergeWithOverlap)
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

TEST(MultidimensionalRange, cantMergeWhenSeparated)
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

TEST(MultidimensionalRange, subtraction)
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
