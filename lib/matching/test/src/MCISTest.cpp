#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/MCIS.h>

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, mcisEmptiness)
{
  MCIS emptyMCIS;
  EXPECT_TRUE(emptyMCIS.empty());

  MCIS nonEmptyMCIS(MultidimensionalRange({
    Range(2, 5),
    Range(3, 7)
  }));

  EXPECT_FALSE(nonEmptyMCIS.empty());
}

TEST(Matching, mcisSize)
{
  MCIS mcis;

  mcis += MultidimensionalRange({
    Range(1, 5),
    Range(3, 7)
  });

  mcis += MultidimensionalRange({
    Range(3, 8),
    Range(2, 5)
  });

  EXPECT_EQ(mcis.size(), 27);
}

TEST(Matching, mcisContainsElement)
{
  MultidimensionalRange range1({
    Range(1, 3),
    Range(4, 7)
  });

  MultidimensionalRange range2({
    Range(5, 9),
    Range(1, 3)
  });

  MCIS mcis({ range1, range2 });

  EXPECT_TRUE(mcis.contains({ 2, 5 }));
  EXPECT_TRUE(mcis.contains({ 5, 1 }));
  EXPECT_FALSE(mcis.contains({ 2, 7 }));
}

TEST(Matching, mcisContainsRange)
{
  MultidimensionalRange range1({
    Range(1, 3),
    Range(4, 7)
  });

  MultidimensionalRange range2({
    Range(5, 9),
    Range(1, 3)
  });

  MCIS mcis({ range1, range2 });

  EXPECT_TRUE(mcis.contains(MultidimensionalRange({
    Range(2, 3),
    Range(5, 6)
  })));

  EXPECT_TRUE(mcis.contains(MultidimensionalRange({
    Range(5, 7),
    Range(1, 3)
  })));

  EXPECT_FALSE(mcis.contains(MultidimensionalRange({
    Range(5, 6),
    Range(5, 7)
  })));
}

TEST(Matching, mcisOverlapsRange)
{
  MultidimensionalRange range1({
    Range(1, 3),
    Range(4, 7)
  });

  MultidimensionalRange range2({
    Range(5, 9),
    Range(1, 3)
  });

  MCIS mcis({ range1, range2 });

  EXPECT_TRUE(mcis.overlaps(MultidimensionalRange({
    Range(2, 4),
    Range(1, 5)
  })));

  EXPECT_TRUE(mcis.overlaps(MultidimensionalRange({
    Range(3, 7),
    Range(2, 4)
  })));

  EXPECT_TRUE(mcis.overlaps(MultidimensionalRange({
    Range(1, 6),
    Range(1, 5)
  })));
}

TEST(Matching, mcisAddRange)
{
  MultidimensionalRange initialRange({
    Range(1, 3),
    Range(4, 7)
  });

  MCIS mcis(initialRange);
  EXPECT_FALSE(mcis.contains({ 2, 3 }));

  MultidimensionalRange additionalRange({
    Range(1, 3),
    Range(2, 4)
  });

  mcis += additionalRange;
  EXPECT_TRUE(mcis.contains({ 2, 3 }));
}

TEST(Matching, mcisAddOverlappingRange)
{
  MultidimensionalRange initialRange({
    Range(1, 3),
    Range(4, 7)
  });

  MCIS mcis(initialRange);

  mcis += MultidimensionalRange({
    Range(1, 9),
    Range(7, 10)
  });

  mcis += MultidimensionalRange({
    Range(7, 9),
    Range(4, 8)
  });

  mcis += MultidimensionalRange({
    Range(2, 8),
    Range(4, 9)
  });

  EXPECT_TRUE(mcis.contains(MultidimensionalRange({
    Range(1, 9),
    Range(4, 10)
  })));
}

TEST(Matching, mcisAddMultipleRanges)
{
  MCIS mcis;

  mcis += MultidimensionalRange({
    Range(3, 5),
    Range(7, 9)
  });

  mcis += MultidimensionalRange({
    Range(3, 5),
    Range(9, 11)
  });

  mcis += MultidimensionalRange({
    Range(5, 8),
    Range(7, 8)
  });

  mcis += MultidimensionalRange({
    Range(5, 8),
    Range(8, 11)
  });

  MultidimensionalRange range({
    Range(3, 8),
    Range(7, 11)
  });

  EXPECT_TRUE(mcis.contains(range));
}

TEST(Matching, mcisRemoveRange)
{
  MultidimensionalRange range({
    Range(2, 5),
    Range(3, 7)
  });

  MCIS original(range);

  MCIS removed(MultidimensionalRange({
    Range(3, 9),
    Range(1, 4)
  }));

  MCIS result = original - removed;

  for (auto indexes : range)
    EXPECT_EQ(result.contains(indexes), !removed.contains(indexes));
}

TEST(Matching, mcisComplement)
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

  MCIS original;

  for (const auto& range : ranges)
    original += range;

  MultidimensionalRange range({
    Range(2, 7),
    Range(0, 6)
  });

  MCIS result = original.complement(range);

  for (auto indexes : ranges[0])
    EXPECT_FALSE(result.contains(indexes));

  for (auto indexes : ranges[1])
    EXPECT_FALSE(result.contains(indexes));

  for (auto indexes : range)
  {
    EXPECT_EQ(result.contains(indexes), std::none_of(ranges.begin(), ranges.end(), [&](const MultidimensionalRange& r) {
      return r.contains(indexes);
    }));
  }
}

TEST(Matching, mcisComplementEmptyBase)
{
  MCIS original;

  MultidimensionalRange range({
    Range(2, 7),
    Range(0, 6)
  });

  MCIS result = original.complement(range);

  for (auto indexes : range)
    EXPECT_TRUE(result.contains(indexes));
}