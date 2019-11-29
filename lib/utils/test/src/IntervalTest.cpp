#include "gtest/gtest.h"
#include <modelica/utils/Interval.hpp>

using namespace modelica;
using namespace std;

TEST(IntervalTest, constructorTest)
{
	IntInterval interval(0, 10);
	EXPECT_EQ(interval.front(), 0);
	EXPECT_EQ(interval.back(), 10);
	EXPECT_EQ(interval.step(), 1);

	interval = IntInterval(0, 10, 2);
	EXPECT_EQ(interval.front(), 0);
	EXPECT_EQ(interval.back(), 10);
	EXPECT_EQ(interval.step(), 2);
}

TEST(IntervalTest, containsTest)
{
	IntInterval interval(0, 10);
	for (int a = 0; a < 10; a++)
		EXPECT_TRUE(interval.contains(0));

	EXPECT_FALSE(interval.contains(10));
	EXPECT_FALSE(interval.contains(-2));
}

TEST(IntervalTest, stepContainTest)
{
	IntInterval interval(3, 10, 2);
	EXPECT_TRUE(interval.contains(3));
	EXPECT_TRUE(interval.contains(5));
	EXPECT_TRUE(interval.contains(5));
	EXPECT_TRUE(interval.contains(9));

	EXPECT_FALSE(interval.contains(10));
	EXPECT_FALSE(interval.contains(4));
	EXPECT_FALSE(interval.contains(-2));
}

TEST(IntervalTest, iteratioInterval)
{
	IntInterval interval(0, 10);
	std::vector<int> values;

	for (int num : interval)
		values.push_back(num);

	EXPECT_EQ(values.size(), 10);

	for (int a = 0; a < 10; a++)
		EXPECT_EQ(values[a], a);
}

TEST(IntervalTest, steppedIntervalIterator)
{
	IntInterval interval(0, 10, 2);
	std::vector<int> values(5);

	std::transform(
			begin(interval), end(interval), begin(values), [](int a) { return a; });

	EXPECT_EQ(values.size(), 5);

	for (int a = 0; a < 5; a++)
		EXPECT_EQ(values[a], a * 2);
}

TEST(IntervalTest, sameStepIntersection)
{
	IntInterval interval(0, 10);
	IntInterval interval2(2, 20);

	EXPECT_FALSE(disjoint(interval, interval2));
	auto result = intersection(interval, interval2);

	EXPECT_EQ(1, result.size());
	EXPECT_EQ(result[0], IntInterval(2, 10));
}

TEST(IntervalTest, multipierOfTest)
{
	for (int a = 1; a < 5; a++)
	{
		EXPECT_TRUE(multipleOf(1, a));
		EXPECT_TRUE(multipleOf(a, 1));
	}

	EXPECT_TRUE(multipleOf(2, 8));
	EXPECT_FALSE(multipleOf(3, 2));
}

TEST(IntervalTest, minCommonTest)
{
	EXPECT_EQ(
			make_pair(3, true), minCommon(IntInterval(3, 4), IntInterval(2, 9)));
	EXPECT_EQ(
			make_pair(3, true), minCommon(IntInterval(3, 4, 4), IntInterval(2, 9)));
	EXPECT_EQ(
			make_pair(0, false), minCommon(IntInterval(3, 4), IntInterval(2, 9, 3)));
	EXPECT_EQ(
			make_pair(5, true), minCommon(IntInterval(3, 6), IntInterval(2, 9, 3)));
}

TEST(IntervalTest, disjointTest)
{
	EXPECT_TRUE(disjoint(IntInterval(3, 10), IntInterval(1, 3)));
	EXPECT_FALSE(disjoint(IntInterval(3, 10), IntInterval(1, 5)));
	EXPECT_TRUE(disjoint(IntInterval(2, 10, 3), IntInterval(1, 5, 2)));
}
TEST(IntervalTest, commonSteppedIntersection)
{
	IntInterval interval(0, 10, 4);
	IntInterval interval2(2, 20, 2);

	EXPECT_FALSE(disjoint(interval, interval2));
	auto result = intersection(interval, interval2);

	EXPECT_EQ(1, result.size());
	EXPECT_EQ(result[0], IntInterval(4, 10, 4));
}

TEST(IntervalTest, intervalUnionTest)
{
	IntInterval interval(1, 5);
	IntInterval interval2(4, 7);

	auto result = intervalUnion(interval, interval2);

	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result[0], IntInterval(1, 7));
}

TEST(IntervalTest, disjoinSingleStepUnionTest)
{
	IntInterval interval(1, 5);
	IntInterval interval2(8, 9);

	auto result = intervalUnion(interval, interval2);

	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0], IntInterval(1, 5));
	EXPECT_EQ(result[1], IntInterval(8, 9));
}

TEST(IntervalTest, steppedUnionTest)
{
	IntInterval interval(1, 5, 3);
	IntInterval interval2(4, 10, 3);

	auto result = intervalUnion(interval, interval2);

	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result[0], IntInterval(1, 10, 3));
}

TEST(IntervalTest, disjoinSteppedUntionTest)
{
	IntInterval interval(1, 5, 3);
	IntInterval interval2(3, 10, 3);

	auto result = intervalUnion(interval, interval2);

	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0], IntInterval(1, 5, 3));
	EXPECT_EQ(result[1], IntInterval(3, 10, 3));
}

TEST(IntervalTest, singleStepIntervalDifference)
{
	IntInterval interval(1, 10);
	IntInterval interval2(4, 6);

	auto result = intervalDifference(interval, interval2);

	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0], IntInterval(1, 4));
	EXPECT_EQ(result[1], IntInterval(6, 10));
}

TEST(IntervalTest, disjointIntervalDifference)
{
	IntInterval interval(1, 4);
	IntInterval interval2(4, 6);

	auto result = intervalDifference(interval, interval2);

	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result[0], IntInterval(1, 4));
}

TEST(IntervalTest, steppedIntervalDifference)
{
	IntInterval interval(1, 8, 2);
	IntInterval interval2(3, 6);

	auto result = intervalDifference(interval, interval2);

	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0].front(), 1);
	EXPECT_EQ(result[1].front(), 7);
}
