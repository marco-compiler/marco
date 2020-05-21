#include "gtest/gtest.h"
#include <vector>

#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

using namespace modelica;
using namespace std;

TEST(IntervalTest, getterTest)
{
	Interval interval(10, 20);

	EXPECT_EQ(10, interval.min());
	EXPECT_EQ(20, interval.max());
}

TEST(IntervalTest, forAllTest)
{
	Interval interval(10, 20);

	vector<size_t> vector;
	for (auto el : interval)
		vector.push_back(el);

	EXPECT_EQ(vector.size(), 10);

	size_t index = 10;
	for (size_t t = 0; t < 10; t++)
		EXPECT_EQ(vector[t], index++);
}

TEST(IntervalTest, containsTest)
{
	Interval interval(10, 20);

	for (size_t t = 0; t < 10; t++)
		EXPECT_FALSE(interval.contains(t));
	for (size_t t = 10; t < 20; t++)
		EXPECT_TRUE(interval.contains(t));
	for (size_t t = 20; t < 30; t++)
		EXPECT_FALSE(interval.contains(t));
}

TEST(IntervalTest, multiDimConstructor)
{
	MultiDimInterval interval({ { 1, 10 }, { 1, 10 } });

	EXPECT_EQ(interval.dimensions(), 2);
}

TEST(IntervalTest, multiIntervalContains)
{
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });

	for (size_t t = 1; t < 10; t++)
		for (size_t z = 1; z < 10; z++)
			EXPECT_TRUE(interval.contains(t, z));
}

TEST(IntervalTest, disjointTest)
{
	Interval i1(1, 10);
	Interval i2(9, 14);
	Interval i3(12, 16);
	EXPECT_FALSE(areDisjoint(i1, i2));
	EXPECT_TRUE(areDisjoint(i3, i1));
	EXPECT_FALSE(areDisjoint(i3, i2));
}

TEST(IntervalTest, multiDimDisjoinTest)
{
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(11, 12), Interval(1, 10) });
	EXPECT_TRUE(areDisjoint(interval, interval2));
}

TEST(IntervalTest, multiDimNonDisjoinTest)
{
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });

	MultiDimInterval interval2({ Interval(6, 12), Interval(1, 10) });
	EXPECT_FALSE(areDisjoint(interval, interval2));
}

TEST(IntervalTest, indexSetUnionTest)
{
	IndexSet set;
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	set.unite(interval);
	EXPECT_EQ(*set.begin(), interval);
}

TEST(IntervalTest, indexSetContainsTest)
{
	IndexSet set;
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	set.unite(interval);
	for (size_t a = 1; a < 10; a++)
		for (size_t b = 1; b < 10; b++)
		{
			EXPECT_TRUE(interval.contains(a, b));
			EXPECT_TRUE(set.contains(a, b));
		}
}

TEST(IntervalTest, multiDimIntersection)
{
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(5, 12), Interval(1, 10) });
	MultiDimInterval result({ Interval(5, 10), Interval(1, 10) });

	EXPECT_EQ(intersection(interval, interval2), result);
}

TEST(IntervalTest, indexSetIntersection)
{
	IndexSet set;
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(5, 12), Interval(1, 10) });
	MultiDimInterval result({ Interval(5, 10), Interval(1, 10) });
	set.unite(interval);

	set.intersecate(interval2);

	EXPECT_EQ(*set.begin(), result);
}

TEST(IntervalTest, indexSetIntersectionTest)
{
	IndexSet set;
	IndexSet set2;
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(5, 12), Interval(1, 10) });
	MultiDimInterval result({ Interval(5, 10), Interval(1, 10) });
	set.unite(interval);
	set2.unite(interval2);

	set.intersecate(set2);

	EXPECT_EQ(*set.begin(), result);
}

TEST(IntervalTest, extendibleFalseTest)
{
	MultiDimInterval interval({ Interval(1, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(5, 12), Interval(1, 10) });
	MultiDimInterval interval3({ Interval(11, 12), Interval(1, 9) });

	EXPECT_FALSE(interval.isExpansionOf(interval2).first);
	EXPECT_FALSE(interval.isExpansionOf(interval3).first);
}

TEST(IntervalTest, extendibleTrueTest)
{
	MultiDimInterval interval({ Interval(3, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(10, 12), Interval(1, 10) });
	MultiDimInterval interval3({ Interval(1, 3), Interval(1, 10) });

	EXPECT_TRUE(interval.isExpansionOf(interval2).first);
	EXPECT_TRUE(interval.isExpansionOf(interval3).first);
}

TEST(IntervalTest, expansion)
{
	MultiDimInterval interval({ Interval(3, 10), Interval(1, 10) });
	MultiDimInterval interval2({ Interval(10, 12), Interval(1, 10) });
	MultiDimInterval result({ Interval(3, 12), Interval(1, 10) });
	interval.expand(interval2);
	EXPECT_EQ(interval, result);
}

TEST(IntervalTest, intervalOrderingTest)
{
	Interval interval(3, 10);
	Interval interval2(2, 10);

	EXPECT_TRUE(interval > interval2);
	EXPECT_TRUE(interval >= interval2);
	EXPECT_FALSE(interval < interval2);
	EXPECT_FALSE(interval <= interval2);
}

TEST(IntervalTest, confrontTest)
{
	MultiDimInterval interval({ Interval(3, 10), Interval(3, 11) });
	MultiDimInterval interval2({ Interval(5, 10), Interval(3, 11) });

	EXPECT_TRUE(interval < interval2);
	EXPECT_TRUE(interval <= interval2);
	EXPECT_FALSE(interval > interval2);
	EXPECT_FALSE(interval >= interval2);
	EXPECT_EQ(interval.confront(interval2), -1);
}

TEST(IntervalTest, singleIntervalDisjointAdjacentTest)
{
	Interval i1(0, 2);
	Interval i2(2, 4);

	EXPECT_TRUE(areDisjoint(i1, i2));
}

TEST(IntervalTest, adjacentDisjointTest)
{
	MultiDimInterval i1({ { 0, 2 }, { 0, 2 } });
	MultiDimInterval i2({ { 2, 4 }, { 0, 2 } });

	EXPECT_TRUE(areDisjoint(i1, i2));
}

TEST(IntervalTest, compactingTest)
{
	IndexSet set;
	set.unite({ Interval(0, 2), Interval(0, 2) });
	set.unite({ Interval(2, 4), Interval(2, 4) });

	EXPECT_EQ(set.partitionsCount(), 2);
	set.unite({ Interval(2, 4), Interval(0, 2) });
	EXPECT_EQ(set.partitionsCount(), 2);
	set.unite({ Interval(0, 2), Interval(2, 4) });
	EXPECT_EQ(set.partitionsCount(), 1);
}

TEST(IntervalTest, singleDimensionReplacementTest)
{
	MultiDimInterval interval({ { 0, 2 }, { 0, 2 } });
	MultiDimInterval result({ { 0, 2 }, { 3, 4 } });
	auto copy = interval.replacedDimension(1, 3, 4);
	EXPECT_EQ(copy, result);
}

TEST(IntervalTest, lineCutsTest)
{
	MultiDimInterval interval({ { 0, 10 }, { 0, 2 } });
	auto result = interval.cutOnDimension(0, { 2, 9 });

	EXPECT_EQ(result[0], MultiDimInterval({ { 0, 2 }, { 0, 2 } }));
	EXPECT_EQ(result[1], MultiDimInterval({ { 2, 9 }, { 0, 2 } }));
	EXPECT_EQ(result[2], MultiDimInterval({ { 9, 10 }, { 0, 2 } }));
}

TEST(IntervalTest, intervalIsFullyContained)
{
	Interval larger(0, 10);
	Interval inside(2, 5);
	Interval notInside(2, 11);

	EXPECT_TRUE(inside.isFullyContained(larger));
	EXPECT_FALSE(notInside.isFullyContained(larger));
}

TEST(IntervalTest, multiDimensionIsFullyContained)
{
	MultiDimInterval larger({ { 0, 10 }, { 0, 10 } });
	MultiDimInterval inside({ { 2, 8 }, { 2, 8 } });
	MultiDimInterval notInside({ { 2, 12 }, { 2, 8 } });

	EXPECT_TRUE(inside.isFullyContained(larger));
	EXPECT_FALSE(notInside.isFullyContained(larger));
}

TEST(IntervalTest, multiDimensionalRemove)
{
	MultiDimInterval larger({ { 0, 10 }, { 0, 10 } });
	MultiDimInterval inside({ { 2, 8 }, { 2, 8 } });

	auto cutted = remove(larger, inside);

	for (size_t a = 0; a < 10; a++)
		for (size_t b = 0; b < 10; b++)
			if ((a < 2 || a >= 8) || (b < 2 || b >= 8))
			{
				EXPECT_TRUE(cutted.contains(a, b));
			}
			else
			{
				EXPECT_FALSE(cutted.contains(a, b));
			}
}

TEST(IntervalTest, indexSetEmptyRemove)
{
	IndexSet set({ { 1, 3 }, { 2, 5 } });
	IndexSet empty;

	IndexSet copy = set;
	set.remove(empty);

	EXPECT_EQ(set, copy);
}

TEST(IntervalTest, fullRemoval)
{
	IndexSet set({ { 0, 5 } });
	IndexSet set2({ { 0, 5 } });

	set.remove(set2);
	EXPECT_EQ(set, IndexSet());
}

TEST(IntervalTest, multiDimensionalfullRemoval)
{
	MultiDimInterval set1({ { 0, 5 } });
	MultiDimInterval set2({ { 0, 5 } });

	auto set = remove(set1, set2);
	EXPECT_EQ(set, IndexSet());
}

TEST(IntervalTest, interavalRageTest)
{
	size_t current = 0;
	MultiDimInterval set1({ { 0, 5 } });

	for (auto val : set1.contentRange())
	{
		EXPECT_EQ(val.size(), 1);
		EXPECT_EQ(current++, val[0]);
	}
	EXPECT_EQ(current, 5);
}

TEST(IntervalTest, interavalRageTestMultiDim)
{
	size_t current = 0;
	MultiDimInterval set1({ { 0, 5 }, { 0, 2 } });

	for (auto val : set1.contentRange())
	{
		EXPECT_EQ(val.size(), 2);
		EXPECT_EQ(current++, (val[0] * 3) + val[1]);
	}
	EXPECT_EQ(current, 17);
}
