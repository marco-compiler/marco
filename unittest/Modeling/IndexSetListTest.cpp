#include "marco/Modeling/IndexSetList.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::impl;

TEST(IndexSetList, empty)
{
  ListIndexSet emptyIndexSet;
  EXPECT_TRUE(emptyIndexSet.empty());

  ListIndexSet nonEmptyIndexSet(MultidimensionalRange({
      Range(2, 5),
      Range(3, 7)
  }));

  EXPECT_FALSE(nonEmptyIndexSet.empty());
}

TEST(IndexSetList, flatSize)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 5),
      Range(3, 7)
  });

  indices += MultidimensionalRange({
      Range(3, 8),
      Range(2, 5)
  });

  EXPECT_EQ(indices.flatSize(), 27);
}

TEST(IndexSetList, clear)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 5),
      Range(3, 7)
  });

  indices.clear();
  EXPECT_EQ(indices.flatSize(), 0);
}

TEST(IndexSetList, containsElement)
{
  MultidimensionalRange range1({
      Range(1, 3),
      Range(4, 7)
  });

  MultidimensionalRange range2({
      Range(5, 9),
      Range(1, 3)
  });

  ListIndexSet indices({range1, range2});

  EXPECT_TRUE(indices.contains(Point({2, 5})));
  EXPECT_TRUE(indices.contains(Point({5, 1})));
  EXPECT_FALSE(indices.contains(Point({2, 7})));
}

TEST(IndexSetList, containsRange)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(4, 7)
  });

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(7, 9)
  });

  indices += MultidimensionalRange({
      Range(5, 9),
      Range(1, 3)
  });

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(2, 3),
      Range(5, 6)
  })));

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(2, 3),
      Range(5, 8)
  })));

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(5, 7),
      Range(1, 3)
  })));

  EXPECT_FALSE(indices.contains(MultidimensionalRange({
      Range(5, 6),
      Range(5, 7)
  })));
}

TEST(IndexSetList, overlapsRange)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(4, 7)
  });

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(7, 9)
  });

  indices += MultidimensionalRange({
      Range(5, 9),
      Range(1, 3)
  });

  EXPECT_TRUE(indices.overlaps(MultidimensionalRange({
      Range(2, 4),
      Range(1, 5)
  })));

  EXPECT_TRUE(indices.overlaps(MultidimensionalRange({
      Range(3, 7),
      Range(2, 4)
  })));

  EXPECT_TRUE(indices.overlaps(MultidimensionalRange({
      Range(1, 6),
      Range(1, 5)
  })));
}

TEST(IndexSetList, addRange)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(4, 7)
  });

  EXPECT_FALSE(indices.contains(Point({2, 3})));

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(2, 4)
  });;

  EXPECT_TRUE(indices.contains(Point({2, 3})));
}

TEST(IndexSetList, addOverlappingRange)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 3),
      Range(4, 7)
  });

  indices += MultidimensionalRange({
      Range(1, 9),
      Range(7, 10)
  });

  indices += MultidimensionalRange({
      Range(7, 9),
      Range(4, 8)
  });

  indices += MultidimensionalRange({
      Range(2, 8),
      Range(4, 9)
  });

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(1, 9),
      Range(4, 10)
  })));
}

TEST(IndexSetList, addMultipleRanges)
{
  ListIndexSet indices;

  indices += MultidimensionalRange({
      Range(3, 5),
      Range(7, 9)
  });

  indices += MultidimensionalRange({
      Range(3, 5),
      Range(9, 11)
  });

  indices += MultidimensionalRange({
      Range(5, 8),
      Range(7, 8)
  });

  indices += MultidimensionalRange({
      Range(5, 8),
      Range(8, 11)
  });

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(3, 8),
      Range(7, 11)
  })));
}

TEST(IndexSetList, removeRange)
{
  MultidimensionalRange range({
      Range(2, 5),
      Range(3, 7)
  });

  ListIndexSet indices(range);

  MultidimensionalRange removed({
      Range(3, 9),
      Range(1, 4)
  });

  indices -= removed;

  for (Point point : range) {
    EXPECT_EQ(indices.contains(point), !removed.contains(point));
  }
}

TEST(IndexSetList, complement)
{
  llvm::SmallVector<MultidimensionalRange, 3> ranges;

  ranges.push_back(MultidimensionalRange({
      Range(1, 5),
      Range(3, 7)
  }));

  ranges.push_back(MultidimensionalRange({
      Range(3, 8),
      Range(2, 5)
  }));

  ListIndexSet original;

  for (const auto& range: ranges) {
    original += range;
  }

  MultidimensionalRange range({
      Range(2, 7),
      Range(0, 6)
  });

  IndexSet result = original.complement(range);

  for (auto indexes: ranges[0])
    EXPECT_FALSE(result.contains(indexes));

  for (auto indexes: ranges[1])
    EXPECT_FALSE(result.contains(indexes));

  for (auto indexes: range) {
    EXPECT_EQ(result.contains(indexes), std::none_of(ranges.begin(), ranges.end(), [&](const MultidimensionalRange& r) {
                return r.contains(indexes);
              }));
  }
}

TEST(IndexSetList, complementEmptyBase)
{
  IndexSet original;

  MultidimensionalRange range({
      Range(2, 7),
      Range(0, 6)
  });

  IndexSet result = original.complement(range);

  for (auto indexes: range)
    EXPECT_TRUE(result.contains(indexes));
}

TEST(IndexSetList, rangesIteration)
{
  MultidimensionalRange a({
      Range(1, 3),
      Range(2, 5),
      Range(8, 10)
  });

  ListIndexSet set(a);

  size_t count = 0;
  for(auto range : llvm::make_range(set.rangesBegin(), set.rangesEnd()))
  {
    EXPECT_EQ(range, a);
    ++count;
  }

  EXPECT_EQ(count, 1);
}

TEST(IndexSetList, indexesIteration)
{
  ListIndexSet range(MultidimensionalRange({
      Range(1, 3),
      Range(2, 5),
      Range(8, 10)
  }));

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
