#include "gtest/gtest.h"
#include "marco/modeling/Point.h"

using namespace ::marco::modeling;

TEST(Point, 1d)
{
  Point p(2);

  EXPECT_EQ(p.rank(), 1);
  EXPECT_EQ(p[0], 2);
}

TEST(Point, 2d)
{
  Point p({2, 5});

  EXPECT_EQ(p.rank(), 2);

  EXPECT_EQ(p[0], 2);
  EXPECT_EQ(p[1], 5);
}

TEST(Point, 3d)
{
  Point p({2, 5, 3});

  EXPECT_EQ(p.rank(), 3);

  EXPECT_EQ(p[0], 2);
  EXPECT_EQ(p[1], 5);
  EXPECT_EQ(p[2], 3);
}

TEST(Point, iteration)
{
  Point p({2, 5, 3});

  size_t counter = 0;
  std::vector<Point::data_type> expected;
  expected.push_back(2);
  expected.push_back(5);
  expected.push_back(3);

  for (const auto& value: p)
    EXPECT_EQ(value, expected[counter++]);

  EXPECT_EQ(counter, 3);
}

TEST(Point, equality)
{
  Point a({2, 5, 3});
  Point b({2, 5, 3});
  Point c({2, 4, 3});
  Point d({2, 5});

  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);

  EXPECT_FALSE(a == c);
  EXPECT_FALSE(c == a);

  EXPECT_FALSE(a == d);
  EXPECT_FALSE(d == a);
}

TEST(Point, inequality)
{
  Point a({2, 5, 3});
  Point b({2, 5, 3});
  Point c({2, 4, 3});
  Point d({2, 5});

  EXPECT_FALSE(a != b);
  EXPECT_FALSE(b != a);

  EXPECT_TRUE(a != c);
  EXPECT_TRUE(c != a);

  EXPECT_TRUE(a != d);
  EXPECT_TRUE(d != a);
}
