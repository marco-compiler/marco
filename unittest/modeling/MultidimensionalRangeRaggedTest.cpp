#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/modeling/MultidimensionalRangeRagged.h"
#include <vector>

using std::vector;
using namespace marco::modeling;

TEST(RaggedValue, addition)
{
  RaggedValue a = {3,4,5};
  RaggedValue b = 1;

  ASSERT_EQ(a+b, RaggedValue({4,5,6}));
}

TEST(RaggedValue, subtraction)
{
  RaggedValue a = {3,4,5};
  RaggedValue b = 1;

  ASSERT_EQ(a-b, RaggedValue({2,3,4}));
}

TEST(RaggedValue, multiplication)
{
  RaggedValue a = {3,4,5};
  RaggedValue b = 2;

  ASSERT_EQ(a*b, RaggedValue({6,8,10}));
}

TEST(RaggedValue, division)
{
  RaggedValue a = {3,4,5};
  RaggedValue b = 2;

  ASSERT_EQ(a/b, RaggedValue({1,2,2}));
}

TEST(RangeRagged, borders)
{
  RangeRagged range(1, 5);

  ASSERT_EQ(range.getBegin(), 1);
  ASSERT_EQ(range.getEnd(), 5);
}

TEST(RangeRagged, size)
{
  RangeRagged range(1, 5);
  ASSERT_EQ(range.size(), 4);
}

TEST(RangeRagged, iteration)
{
  RangeRagged range(1, 5);

  auto begin = range.begin();
  auto end = range.end();

  auto value = range.getBegin();

  for (auto it = begin; it != end; ++it)
    EXPECT_EQ(RaggedValue(*it), value++);
}

TEST(RangeRagged, containsValue)
{
  RangeRagged range(3, 6);

  EXPECT_FALSE(range.contains(2));
  EXPECT_TRUE(range.contains(3));
  EXPECT_TRUE(range.contains(4));
  EXPECT_TRUE(range.contains(5));
  EXPECT_FALSE(range.contains(6));
}

TEST(RangeRagged, containsRange)
{
  RangeRagged range(3, 6);

  EXPECT_FALSE(range.contains(RangeRagged(1, 2)));
  EXPECT_FALSE(range.contains(RangeRagged(1, 3)));
  EXPECT_FALSE(range.contains(RangeRagged(1, 4)));
  EXPECT_TRUE(range.contains(RangeRagged(3, 5)));
  EXPECT_TRUE(range.contains(RangeRagged(4, 6)));
  EXPECT_TRUE(range.contains(RangeRagged(3, 6)));
  EXPECT_FALSE(range.contains(RangeRagged(5, 7)));
  EXPECT_FALSE(range.contains(RangeRagged(6, 9)));
  EXPECT_FALSE(range.contains(RangeRagged(7, 9)));
}

TEST(RangeRagged, overlap)
{
  RangeRagged x(1, 5);
  RangeRagged y(2, 7);

  EXPECT_TRUE(x.overlaps(y));
  EXPECT_TRUE(y.overlaps(x));
}

TEST(RangeRagged, touchingBordersDoNotOverlap)
{
  RangeRagged x(1, 5);
  RangeRagged y(5, 7);

  EXPECT_FALSE(x.overlaps(y));
  EXPECT_FALSE(y.overlaps(x));
}

TEST(RangeRagged, merge)
{
  RangeRagged x(1, 5);

  // Overlapping
  RangeRagged y(3, 11);

  EXPECT_TRUE(x.canBeMerged(y));
  EXPECT_TRUE(y.canBeMerged(x));

  EXPECT_EQ(x.merge(y).getBegin(), 1);
  EXPECT_EQ(x.merge(y).getEnd(), 11);

  EXPECT_EQ(y.merge(x).getBegin(), 1);
  EXPECT_EQ(y.merge(x).getEnd(), 11);

  // Touching borders
  RangeRagged z(5, 7);

  EXPECT_TRUE(x.canBeMerged(z));
  EXPECT_TRUE(z.canBeMerged(x));

  EXPECT_EQ(x.merge(z).getBegin(), 1);
  EXPECT_EQ(x.merge(z).getEnd(), 7);

  EXPECT_EQ(z.merge(x).getBegin(), 1);
  EXPECT_EQ(z.merge(x).getEnd(), 7);
}

TEST(RangeRagged, subtraction)
{
  RangeRagged a(3, 7);

  // Overlapping
  RangeRagged b(5, 11);

  EXPECT_THAT(a.subtract(b), testing::UnorderedElementsAre(RangeRagged(3, 5)));
  EXPECT_THAT(b.subtract(a), testing::UnorderedElementsAre(RangeRagged(7, 11)));

  // Fully contained
  RangeRagged c(2, 11);
  RangeRagged d(2, 5);
  RangeRagged e(7, 11);

  EXPECT_THAT(a.subtract(c), testing::IsEmpty());
  EXPECT_THAT(c.subtract(a), testing::UnorderedElementsAre(RangeRagged(2, 3), RangeRagged(7, 11)));
  EXPECT_THAT(c.subtract(d), testing::UnorderedElementsAre(RangeRagged(5, 11)));
  EXPECT_THAT(c.subtract(e), testing::UnorderedElementsAre(RangeRagged(2, 7)));
}

TEST(RangeRagged, subtractionRagged)
{
//todo
}


//multidim

TEST(MultidimensionalRangeRagged, rank)
{
  MultidimensionalRangeRagged range({
      RangeRagged(1, 3),
      RangeRagged(2, 5),
      RangeRagged(7, 10)
  });

  EXPECT_EQ(range.rank(), 3);
}

TEST(MultidimensionalRangeRagged, flatSize)
{
  MultidimensionalRangeRagged range({
      RangeRagged(1, 3),
      RangeRagged(2, 5),
      RangeRagged(7, 10)
  });

  EXPECT_EQ(range.flatSize(), 18);
}

TEST(MultidimensionalRangeRagged, iteration)
{
  MultidimensionalRangeRagged range({
      RangeRagged(1, 3),
      RangeRagged(2, 5),
      RangeRagged(8, 10)
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

  long index = 0;
  
  auto indexes = range;//.contentRange();

  for (auto it = indexes.begin(), end = indexes.end(); it != end; ++it) {
    auto values = *it;

    EXPECT_EQ(values.rank(), 3);
    EXPECT_EQ(values[0], std::get<0>(expected[index]));
    EXPECT_EQ(values[1], std::get<1>(expected[index]));
    EXPECT_EQ(values[2], std::get<2>(expected[index]));

    ++index;
  }
}

TEST(MultidimensionalRangeRagged, iterationRagged)
{
  MultidimensionalRangeRagged range({
      RangeRagged(1, 4),
      RangeRagged({RangeRagged(0,3),RangeRagged(1,5),RangeRagged(3,4)}),
      RangeRagged(3, 5)
  });

  llvm::SmallVector<std::tuple<long, long, long>, 3> expected;
  expected.emplace_back(1, 0, 3);
  expected.emplace_back(1, 0, 4);
  expected.emplace_back(1, 1, 3);
  expected.emplace_back(1, 1, 4);
  expected.emplace_back(1, 2, 3);
  expected.emplace_back(1, 2, 4);
  
  expected.emplace_back(2, 1, 3);
  expected.emplace_back(2, 1, 4);
  expected.emplace_back(2, 2, 3);
  expected.emplace_back(2, 2, 4);
  expected.emplace_back(2, 3, 3);
  expected.emplace_back(2, 3, 4);
  expected.emplace_back(2, 4, 3);
  expected.emplace_back(2, 4, 4);

  expected.emplace_back(3, 3, 3);
  expected.emplace_back(3, 3, 4);

  long index = 0;
  
  auto indexes = range;//.contentRange();

  for (auto it = indexes.begin(), end = indexes.end(); it != end; ++it) {
    auto values = *it;

    EXPECT_EQ(values.rank(), 3);
    EXPECT_EQ(values[0], std::get<0>(expected[index]));
    EXPECT_EQ(values[1], std::get<1>(expected[index]));
    EXPECT_EQ(values[2], std::get<2>(expected[index]));

    ++index;
  }
}

TEST(MultidimensionalRangeRagged, overlap)
{
  MultidimensionalRangeRagged x({
      RangeRagged(1, 3),
      RangeRagged(2, 4)
  });

  MultidimensionalRangeRagged y({
      RangeRagged(2, 3),
      RangeRagged(3, 5)
  });

  EXPECT_TRUE(x.overlaps(y));
  EXPECT_TRUE(y.overlaps(x));
}

TEST(MultidimensionalRangeRagged, touchingBordersDoNotOverlap)
{
  MultidimensionalRangeRagged x({
      RangeRagged(1, 3),
      RangeRagged(2, 4)
  });

  MultidimensionalRangeRagged y({
      RangeRagged(3, 5),
      RangeRagged(3, 5)
  });

  EXPECT_FALSE(x.overlaps(y));
  EXPECT_FALSE(y.overlaps(x));
}

TEST(MultidimensionalRangeRagged, canMergeWithTouchingBorders)
{
  MultidimensionalRangeRagged x({
      RangeRagged(1, 3),
      RangeRagged(2, 4)
  });

  MultidimensionalRangeRagged y({
      RangeRagged(1, 3),
      RangeRagged(4, 7)
  });

  EXPECT_TRUE(x.canBeMerged(y).first);
  EXPECT_TRUE(y.canBeMerged(x).first);

  MultidimensionalRangeRagged z = x.merge(y, x.canBeMerged(y).second);

  EXPECT_EQ(z[0].getBegin(), 1);
  EXPECT_EQ(z[0].getEnd(), 3);
  EXPECT_EQ(z[1].getBegin(), 2);
  EXPECT_EQ(z[1].getEnd(), 7);

  MultidimensionalRangeRagged t = x.merge(y, x.canBeMerged(y).second);
  EXPECT_EQ(z, t);
}

TEST(MultidimensionalRangeRagged, canMergeWithOverlap)
{
  MultidimensionalRangeRagged x({
      RangeRagged(1, 3),
      RangeRagged(2, 6)
  });

  MultidimensionalRangeRagged y({
      RangeRagged(1, 3),
      RangeRagged(4, 7)
  });

  EXPECT_TRUE(x.canBeMerged(y).first);
  EXPECT_TRUE(y.canBeMerged(x).first);

  MultidimensionalRangeRagged z = x.merge(y, x.canBeMerged(y).second);

  EXPECT_EQ(z[0].getBegin(), 1);
  EXPECT_EQ(z[0].getEnd(), 3);
  EXPECT_EQ(z[1].getBegin(), 2);
  EXPECT_EQ(z[1].getEnd(), 7);

  MultidimensionalRangeRagged t = x.merge(y, x.canBeMerged(y).second);
  EXPECT_EQ(z, t);
}

TEST(MultidimensionalRangeRagged, cantMergeWhenSeparated)
{
  MultidimensionalRangeRagged x({
      RangeRagged(1, 3),
      RangeRagged(2, 4)
  });

  MultidimensionalRangeRagged y({
      RangeRagged(3, 5),
      RangeRagged(4, 7)
  });

  EXPECT_FALSE(x.canBeMerged(y).first);
  EXPECT_FALSE(y.canBeMerged(x).first);
}

TEST(MultidimensionalRangeRagged, subtraction)
{
  MultidimensionalRangeRagged a({
      RangeRagged(2, 10),
      RangeRagged(3, 7),
      RangeRagged(1, 8)
  });

  // Fully contained in 'a'
  MultidimensionalRangeRagged b({
      RangeRagged(6, 8),
      RangeRagged(4, 6),
      RangeRagged(3, 7)
  });

  EXPECT_THAT(a.subtract(b), testing::UnorderedElementsAre(
      MultidimensionalRangeRagged({
          RangeRagged(2, 6),
          RangeRagged(3, 7),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(8, 10),
          RangeRagged(3, 7),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(3, 4),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(6, 7),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(4, 6),
          RangeRagged(1, 3)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(4, 6),
          RangeRagged(7, 8)
      })));

  // 2 dimensions fully contained, 1 fully traversing
  MultidimensionalRangeRagged c({
      RangeRagged(6, 8),
      RangeRagged(1, 9),
      RangeRagged(3, 7)
  });

  EXPECT_THAT(a.subtract(c), testing::UnorderedElementsAre(
      MultidimensionalRangeRagged({
          RangeRagged(2, 6),
          RangeRagged(3, 7),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(8, 10),
          RangeRagged(3, 7),
          RangeRagged(1, 8)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(3, 7),
          RangeRagged(1, 3)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(6, 8),
          RangeRagged(3, 7),
          RangeRagged(7, 8)
      })));

  // 1 dimension fully contained, 2 fully traversing
  MultidimensionalRangeRagged d({
      RangeRagged(1, 15),
      RangeRagged(1, 9),
      RangeRagged(3, 7)
  });

  EXPECT_THAT(a.subtract(d), testing::UnorderedElementsAre(
      MultidimensionalRangeRagged({
          RangeRagged(2, 10),
          RangeRagged(3, 7),
          RangeRagged(1, 3)
      }),
      MultidimensionalRangeRagged({
          RangeRagged(2, 10),
          RangeRagged(3, 7),
          RangeRagged(7, 8)
      })));

  // 3 dimensions fully traversing
  MultidimensionalRangeRagged e({
      RangeRagged(1, 15),
      RangeRagged(1, 9),
      RangeRagged(0, 11)
  });

  EXPECT_THAT(a.subtract(e), testing::IsEmpty());
}


// old interval tests
TEST(MultidimensionalRangeRagged, getterTest)
{
	RangeRagged interval(10, 20);

	EXPECT_EQ(10, interval.min());
	EXPECT_EQ(20, interval.max());
}

TEST(MultidimensionalRangeRagged, forAllTest)
{
	RangeRagged interval(10, 20);

	vector<long> vector;
	for (auto el : interval)
		vector.push_back(el);

	EXPECT_EQ(vector.size(), 10);

	long index = 10;
	for (long t = 0; t < 10; t++)
		EXPECT_EQ(vector[t], index++);
}

TEST(MultidimensionalRangeRagged, containsTest)
{
	RangeRagged interval(10, 20);

	for (long t = 0; t < 10; t++)
		EXPECT_FALSE(interval.contains(t));
	for (long t = 10; t < 20; t++)
		EXPECT_TRUE(interval.contains(t));
	for (long t = 20; t < 30; t++)
		EXPECT_FALSE(interval.contains(t));
}

TEST(MultidimensionalRangeRagged, multiDimConstructor)
{
	MultidimensionalRangeRagged interval({ { 1, 10 }, { 1, 10 } });

	EXPECT_EQ(interval.dimensions(), 2);
}

TEST(MultidimensionalRangeRagged, multiIntervalContains)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });

	for (long t = 1; t < 10; t++){
		EXPECT_FALSE(interval.contains({t, 0}));
		for (long z = 1; z < 10; z++){
			EXPECT_TRUE(interval.contains({t, z}));
		}
		for (long z = 10; z < 15; z++){
			EXPECT_FALSE(interval.contains({t, z}));
		}
	}
	for (long z = 10; z < 15; z++){
		EXPECT_FALSE(interval.contains({z,0}));
	}
}
TEST(MultidimensionalRangeRagged, multiRangeRaggedContains)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });

	for (long t = 1; t < 10; t++){
		EXPECT_FALSE(interval.contains({t, 0}));
		for (long z = 1; z < 10; z++){
			EXPECT_TRUE(interval.contains({t, z}));
		}
		for (long z = 10; z < 15; z++){
			EXPECT_FALSE(interval.contains({t, z}));
		}
	}
	for (long z = 10; z < 15; z++){
		EXPECT_FALSE(interval.contains({z,0}));
	}

	interval = MultidimensionalRangeRagged({ {0,2}, {{0,2},{0,3}} });

	EXPECT_TRUE(interval.contains({0,0}));
	EXPECT_TRUE(interval.contains({0,1}));
	EXPECT_FALSE(interval.contains({0,2}));
	EXPECT_TRUE(interval.contains({1,0}));
	EXPECT_TRUE(interval.contains({1,1}));
	EXPECT_TRUE(interval.contains({1,2}));
}

TEST(MultidimensionalRangeRagged, RaggedIntervalsContains)
{
	auto interval = MultidimensionalRangeRagged({ {0,2}, {{0,2},{0,3}} });
	EXPECT_TRUE(interval.contains(interval));
	// {0,0},{0,1}
	// {1,0},{1,1},{1,2}

	MultidimensionalRangeRagged other ={ {0,2}, {1,2} };

	EXPECT_TRUE(interval.contains(other));
	EXPECT_FALSE(other.contains(interval));

	other = MultidimensionalRangeRagged({ {0,2}, {0,3} });

	EXPECT_FALSE(interval.contains(other));
	EXPECT_TRUE(other.contains(interval));	

	interval = MultidimensionalRangeRagged({ {0,2}, {{0,2},{0,3}}, {0,3}});
	other    = MultidimensionalRangeRagged({ {0,2}, {0,2},         {{0,2},{0,3}} });

	EXPECT_TRUE(interval.contains(other));
	EXPECT_FALSE(other.contains(interval));

	other    = MultidimensionalRangeRagged({ {0,2}, {0,2},         {{0,4},{0,3}} });

	EXPECT_FALSE(interval.contains(other));
	EXPECT_FALSE(other.contains(interval));	

	interval = MultidimensionalRangeRagged({ {0,2}, {{0,2},{0,3}}, {{0,4},{0,3}} });

	EXPECT_TRUE(interval.contains(other));
	EXPECT_FALSE(other.contains(interval));	

	interval = MultidimensionalRangeRagged({ {0,2}, {{0,2},{0,3}}, {{0,4},{{0,3},{0,4},{0,3}}} });

	EXPECT_TRUE(interval.contains(other));
	EXPECT_FALSE(other.contains(interval));	


}


TEST(MultidimensionalRangeRagged, disjointTest)
{
	RangeRagged i1(1, 10);
	RangeRagged i2(9, 14);
	RangeRagged i3(12, 16);
	EXPECT_TRUE(i1.overlaps(i2));
	EXPECT_FALSE(i3.overlaps(i1));
	EXPECT_TRUE(i3.overlaps(i2));
}


TEST(MultidimensionalRangeRagged, multiDimDisjoinTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(11, 12), RangeRagged(1, 10) });
	EXPECT_FALSE(interval.overlaps(interval2));
}

TEST(MultidimensionalRangeRagged, multiDimNonDisjoinTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });

	MultidimensionalRangeRagged interval2({ RangeRagged(6, 12), RangeRagged(1, 10) });
	
	EXPECT_TRUE(interval.overlaps(interval2));
}

TEST(MultidimensionalRangeRagged, multiDimRaggedDisjoinTest)
{
	MultidimensionalRangeRagged interval({ {1, 10}, {1, 10} });
	MultidimensionalRangeRagged interval2({ {11, 12}, {1, 10} });
	EXPECT_FALSE(interval.overlaps(interval2));

	interval = MultidimensionalRangeRagged({ {1, 10}, {1, 10} });
	interval2 = MultidimensionalRangeRagged({ {2, 5}, {11, 20} });
	EXPECT_FALSE(interval.overlaps(interval2));


	interval  = MultidimensionalRangeRagged({ {0, 2}, {{0,2},{0,3}} });
	interval2 = MultidimensionalRangeRagged({ {0, 2}, {{3,4},{3,20}} });	
	EXPECT_FALSE(interval.overlaps(interval2));

	interval  = MultidimensionalRangeRagged({ {0, 2},  {{0,5},{0,5}} , 	 {0,3} 		});
	interval2 = MultidimensionalRangeRagged({ {0, 2}, {{5,10},{4,20}}, {{4,6},{6,8}} });
	EXPECT_FALSE(interval.overlaps(interval2));
}

TEST(MultidimensionalRangeRagged, multiDimRaggedNonDisjoinTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(6, 12), RangeRagged(1, 10) });
	EXPECT_TRUE(interval.overlaps(interval2));

	interval  = MultidimensionalRangeRagged({ {0, 2}, {{0,5},{0,3}} });
	interval2 = MultidimensionalRangeRagged({ {0, 2}, {{5,10},{2,20}} });
	EXPECT_TRUE(interval.overlaps(interval2));

	interval  = MultidimensionalRangeRagged({ {0, 2},  {{0,5},{0,5}} , 	 {0,3} 		});
	interval2 = MultidimensionalRangeRagged({ {0, 2}, {{5,10},{4,20}}, {{4,6},{2,8}} });
	EXPECT_TRUE(interval.overlaps(interval2));
}

// TEST(MultidimensionalRangeRagged, indexSetUnionTest)
// {
// 	IndexSet set;
// 	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
// 	set.unite(interval);
// 	EXPECT_EQ(*set.begin(), interval);
// }

// TEST(MultidimensionalRangeRagged, indexSetContainsTest)
// {
// 	IndexSet set;
// 	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
// 	set.unite(interval);
// 	for (long a = 1; a < 10; a++)
// 		for (long b = 1; b < 10; b++)
// 		{
// 			EXPECT_TRUE(interval.contains(a, b));
// 			EXPECT_TRUE(set.contains(a, b));
// 		}
// }

TEST(MultidimensionalRangeRagged, multiDimIntersection)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(5, 12), RangeRagged(1, 10) });
	MultidimensionalRangeRagged result({ RangeRagged(5, 10), RangeRagged(1, 10) });

	EXPECT_EQ(interval.intersect(interval2), result);
}

TEST(MultidimensionalRangeRagged, multiDimRaggedIntersection)
{
	{
		MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
		MultidimensionalRangeRagged interval2({ RangeRagged(5, 12), RangeRagged(1, 10) });
		MultidimensionalRangeRagged result({ RangeRagged(5, 10), RangeRagged(1, 10) });

		EXPECT_EQ(interval.intersect(interval2), result);
	}
	{
		MultidimensionalRangeRagged interval({ {0,2}, {{0,5},{2,6}} });
		MultidimensionalRangeRagged interval2({ {1,4}, {1,5} });
		MultidimensionalRangeRagged result({ {1,2}, {2,5} });

		EXPECT_EQ(interval.intersect(interval2), result);
	}
	{
		MultidimensionalRangeRagged interval( {{1,4},{ {1,10}, {5,10}, {3,8} },{1,5} });
		MultidimensionalRangeRagged interval2({{2,4},{ {2,6},  {3, 7} 	    },{ {{2,10},{1,3},{2,4},{3,5}}, {1,4}}});
		MultidimensionalRangeRagged result(	 {{2,4},{ {5,6},{3,7}		    },{ {3,5},{1,4}}});

		EXPECT_EQ(interval.intersect(interval2), result);
	}
	
}	

// TEST(MultidimensionalRangeRagged, indexSetIntersection)
// {
// 	IndexSet set;
// 	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
// 	MultidimensionalRangeRagged interval2({ RangeRagged(5, 12), RangeRagged(1, 10) });
// 	MultidimensionalRangeRagged result({ RangeRagged(5, 10), RangeRagged(1, 10) });
// 	set.unite(interval);

// 	set.intersecate(interval2);

// 	EXPECT_EQ(*set.begin(), result);
// }

// TEST(MultidimensionalRangeRagged, indexSetIntersectionTest)
// {
// 	IndexSet set;
// 	IndexSet set2;
// 	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
// 	MultidimensionalRangeRagged interval2({ RangeRagged(5, 12), RangeRagged(1, 10) });
// 	MultidimensionalRangeRagged result({ RangeRagged(5, 10), RangeRagged(1, 10) });
// 	set.unite(interval);
// 	set2.unite(interval2);

// 	set.intersecate(set2);

// 	EXPECT_EQ(*set.begin(), result);
// }

TEST(MultidimensionalRangeRagged, extendibleFalseTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(1, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(5, 12), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval3({ RangeRagged(11, 12), RangeRagged(1, 9) });

	EXPECT_TRUE(interval.canBeMerged(interval2).first);//{1,12},{1,10}
	EXPECT_FALSE(interval.canBeMerged(interval3).first);
}

TEST(MultidimensionalRangeRagged, extendibleTrueTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(3, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(10, 12), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval3({ RangeRagged(1, 3), RangeRagged(1, 10) });

	EXPECT_TRUE(interval.canBeMerged(interval2).first);
	EXPECT_TRUE(interval.canBeMerged(interval3).first);
}

TEST(MultidimensionalRangeRagged, expansion)
{
	MultidimensionalRangeRagged interval({ RangeRagged(3, 10), RangeRagged(1, 10) });
	MultidimensionalRangeRagged interval2({ RangeRagged(10, 12), RangeRagged(1, 10) });
	MultidimensionalRangeRagged result({ RangeRagged(3, 12), RangeRagged(1, 10) });
	interval = interval.merge(interval2,0);
	EXPECT_EQ(interval, result);
}

TEST(MultidimensionalRangeRagged, intervalOrderingTest)
{
	RangeRagged interval(3, 10);
	RangeRagged interval2(2, 10);

	EXPECT_TRUE(interval > interval2);
	EXPECT_TRUE(interval >= interval2);
	EXPECT_FALSE(interval < interval2);
	EXPECT_FALSE(interval <= interval2);
}

TEST(MultidimensionalRangeRagged, confrontTest)
{
	MultidimensionalRangeRagged interval({ RangeRagged(3, 10), RangeRagged(3, 11) });
	MultidimensionalRangeRagged interval2({ RangeRagged(5, 10), RangeRagged(3, 11) });

	EXPECT_TRUE(interval < interval2);
	// EXPECT_TRUE(interval <= interval2);
	EXPECT_FALSE(interval > interval2);
	// EXPECT_FALSE(interval >= interval2);
	// EXPECT_EQ(interval.confront(interval2), -1);
}

TEST(MultidimensionalRangeRagged, singleIntervalDisjointAdjacentTest)
{
	RangeRagged i1(0, 2);
	RangeRagged i2(2, 4);

	EXPECT_FALSE(i1.overlaps(i2));
}

TEST(MultidimensionalRangeRagged, adjacentDisjointTest)
{
	MultidimensionalRangeRagged i1({ { 0, 2 }, { 0, 2 } });
	MultidimensionalRangeRagged i2({ { 2, 4 }, { 0, 2 } });

	EXPECT_FALSE(i1.overlaps(i2));
}

// TEST(MultidimensionalRangeRagged, compactingTest)
// {
// 	IndexSet set;
// 	set.unite({ RangeRagged(0, 2), RangeRagged(0, 2) });
// 	set.unite({ RangeRagged(2, 4), RangeRagged(2, 4) });

// 	EXPECT_EQ(set.partitionsCount(), 2);
// 	set.unite({ RangeRagged(2, 4), RangeRagged(0, 2) });
// 	EXPECT_EQ(set.partitionsCount(), 2);
// 	set.unite({ RangeRagged(0, 2), RangeRagged(2, 4) });
// 	EXPECT_EQ(set.partitionsCount(), 1);
// }



TEST(MultidimensionalRangeRagged, intervalIsFullyContained)
{
	RangeRagged larger(0, 10);
	RangeRagged inside(2, 5);
	RangeRagged notInside(2, 11);

	EXPECT_TRUE(inside.contains(inside));
	EXPECT_TRUE(larger.contains(inside));
	EXPECT_FALSE(larger.contains(notInside));
}

TEST(MultidimensionalRangeRagged, multiDimensionIsFullyContained)
{
	MultidimensionalRangeRagged larger({ { 0, 10 }, { 0, 10 } });
	MultidimensionalRangeRagged inside({ { 2, 8 }, { 2, 8 } });
	MultidimensionalRangeRagged notInside({ { 2, 12 }, { 2, 8 } });

	EXPECT_TRUE(inside.contains(inside));
	EXPECT_TRUE(larger.contains(inside));
	EXPECT_FALSE(larger.contains(notInside));
}

// TEST(MultidimensionalRangeRagged, multiDimensionalRemove)
// {
// 	MultidimensionalRangeRagged larger({ { 0, 10 }, { 0, 10 } });
// 	MultidimensionalRangeRagged inside({ { 2, 8 }, { 2, 8 } });

// 	auto cutted = remove(larger, inside);

// 	for (long a = 0; a < 10; a++)
// 		for (long b = 0; b < 10; b++)
// 			if ((a < 2 || a >= 8) || (b < 2 || b >= 8))
// 			{
// 				EXPECT_TRUE(cutted.contains(a, b));
// 			}
// 			else
// 			{
// 				EXPECT_FALSE(cutted.contains(a, b));
// 			}
// }

// TEST(MultidimensionalRangeRagged, indexSetEmptyRemove)
// {
// 	IndexSet set({ { 1, 3 }, { 2, 5 } });
// 	IndexSet empty;

// 	IndexSet copy = set;
// 	set.remove(empty);

// 	EXPECT_EQ(set, copy);
// }

// TEST(MultidimensionalRangeRagged, fullRemoval)
// {
// 	IndexSet set({ { 0, 5 } });
// 	IndexSet set2({ { 0, 5 } });

// 	set.remove(set2);
// 	EXPECT_EQ(set, IndexSet());
// }

// TEST(MultidimensionalRangeRagged, multiDimensionalfullRemoval)
// {
// 	MultidimensionalRangeRagged set1({ { 0, 5 } });
// 	MultidimensionalRangeRagged set2({ { 0, 5 } });

// 	auto set = remove(set1, set2);
// 	EXPECT_EQ(set, IndexSet());
// }

TEST(MultidimensionalRangeRagged, interavalRageTest)
{
	long current = 0;
	MultidimensionalRangeRagged set1({ { 0, 5 } });

	for (auto val : set1)
	{
		EXPECT_EQ(val.rank(), 1);
		EXPECT_EQ(current++, val[0]);
	}
	EXPECT_EQ(current, 5);
}

TEST(MultidimensionalRangeRagged, interavalRageTestMultiDim)
{
	long current = 0;
	MultidimensionalRangeRagged set1({ { 0, 5 }, { 0, 2 } });

	for (auto val : set1)
	{
		EXPECT_EQ(val.rank(), 2);
		EXPECT_EQ(current++, (val[0] * 2) + val[1]);
	}
	EXPECT_EQ(current, 10);
}
