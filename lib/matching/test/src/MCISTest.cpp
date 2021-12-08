#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/matching/MCIS.h>

using namespace marco::matching;

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

  mcis.add(additionalRange);
  EXPECT_TRUE(mcis.contains({ 2, 3 }));
}

TEST(Matching, mcisAddOverlappingRange)
{
  MultidimensionalRange initialRange({
    Range(1, 3),
    Range(4, 7)
  });

  MCIS mcis(initialRange);

  mcis.add(MultidimensionalRange({
    Range(1, 9),
    Range(7, 10)
  }));

  mcis.add(MultidimensionalRange({
    Range(7, 9),
    Range(4, 8)
  }));

  mcis.add(MultidimensionalRange({
    Range(2, 8),
    Range(4, 9)
  }));

  EXPECT_TRUE(mcis.contains(MultidimensionalRange({
    Range(1, 9),
    Range(4, 10)
  })));
}

TEST(Matching, mcisAddMultipleRanges)
{
  MCIS mcis;

  mcis.add(MultidimensionalRange({
    Range(3, 5),
    Range(7, 9)
  }));

  mcis.add(MultidimensionalRange({
    Range(3, 5),
    Range(9, 11)
  }));

  mcis.add(MultidimensionalRange({
    Range(5, 8),
    Range(7, 8)
  }));

  mcis.add(MultidimensionalRange({
    Range(5, 8),
    Range(8, 11)
  }));

  MultidimensionalRange range({
    Range(3, 8),
    Range(7, 11)
  });

  EXPECT_TRUE(mcis.contains(range));
}