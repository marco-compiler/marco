#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/Modeling/IndexSet.h"

using namespace ::marco::modeling;

TEST(IndexSet, empty)
{
  IndexSet emptyIndexSet;
  EXPECT_TRUE(emptyIndexSet.empty());

  IndexSet nonEmptyIndexSet(MultidimensionalRange({
      Range(2, 5),
      Range(3, 7)
  }));

  EXPECT_FALSE(nonEmptyIndexSet.empty());
}

TEST(IndexSet, size)
{
  IndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 5),
      Range(3, 7)
  });

  indices += MultidimensionalRange({
      Range(3, 8),
      Range(2, 5)
  });

  EXPECT_EQ(indices.size(), 27);
}

TEST(IndexSet, clear)
{
  IndexSet indices;

  indices += MultidimensionalRange({
      Range(1, 5),
      Range(3, 7)
  });

  indices.clear();
  EXPECT_EQ(indices.size(), 0);
}

TEST(IndexSet, containsElement)
{
  MultidimensionalRange range1({
      Range(1, 3),
      Range(4, 7)
  });

  MultidimensionalRange range2({
      Range(5, 9),
      Range(1, 3)
  });

  IndexSet indices({range1, range2});

  EXPECT_TRUE(indices.contains({2, 5}));
  EXPECT_TRUE(indices.contains({5, 1}));
  EXPECT_FALSE(indices.contains({2, 7}));
}

TEST(IndexSet, containsRange)
{
  MultidimensionalRange range1({
      Range(1, 3),
      Range(4, 7)
  });

  MultidimensionalRange range2({
      Range(5, 9),
      Range(1, 3)
  });

  IndexSet indices({range1, range2});

  EXPECT_TRUE(indices.contains(MultidimensionalRange({
      Range(2, 3),
      Range(5, 6)
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

TEST(IndexSet, overlapsRange)
{
  MultidimensionalRange range1({
      Range(1, 3),
      Range(4, 7)
  });

  MultidimensionalRange range2({
      Range(5, 9),
      Range(1, 3)
  });

  IndexSet indices({range1, range2});

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

TEST(IndexSet, addRange)
{
  MultidimensionalRange initialRange({
      Range(1, 3),
      Range(4, 7)
  });

  IndexSet indices(initialRange);
  EXPECT_FALSE(indices.contains({2, 3}));

  MultidimensionalRange additionalRange({
      Range(1, 3),
      Range(2, 4)
  });

  indices += additionalRange;
  EXPECT_TRUE(indices.contains({2, 3}));
}

TEST(IndexSet, addOverlappingRange)
{
  MultidimensionalRange initialRange({
      Range(1, 3),
      Range(4, 7)
  });

  IndexSet indices(initialRange);

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

TEST(IndexSet, addMultipleRanges)
{
  IndexSet indices;

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

  MultidimensionalRange range({
      Range(3, 8),
      Range(7, 11)
  });

  EXPECT_TRUE(indices.contains(range));
}

TEST(IndexSet, removeRange)
{
  MultidimensionalRange range({
      Range(2, 5),
      Range(3, 7)
  });

  IndexSet original(range);

  IndexSet removed(MultidimensionalRange({
      Range(3, 9),
      Range(1, 4)
  }));

  IndexSet result = original - removed;

  for (auto indexes: range) {
    EXPECT_EQ(result.contains(indexes), !removed.contains(indexes));
  }
}

TEST(IndexSet, complement)
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

  IndexSet original;

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

TEST(IndexSet, complementEmptyBase)
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