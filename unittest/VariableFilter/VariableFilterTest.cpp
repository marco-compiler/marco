#include "marco/VariableFilter/VariableFilter.h"
#include "gtest/gtest.h"

using namespace marco;

TEST(VariableFilter, scalarVariable) {
  auto str = R"(x.y)";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x.y", 0);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());
  EXPECT_TRUE(filters[0].getRanges().empty());
}

TEST(VariableFilter, unboundedArray) {
  auto str = R"(x.y[$:$])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x.y", 1);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 1);
  EXPECT_FALSE(ranges[0].hasLowerBound());
  EXPECT_FALSE(ranges[0].hasUpperBound());
}

TEST(VariableFilter, arrayWithLowerBound) {
  auto str = R"(x.y[1:$])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x.y", 1);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 1);
  EXPECT_TRUE(ranges[0].hasLowerBound());
  EXPECT_EQ(ranges[0].getLowerBound(), 1);
  EXPECT_FALSE(ranges[0].hasUpperBound());
}

TEST(VariableFilter, arrayWithUpperBound) {
  auto str = R"(x.y[$:3])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x.y", 1);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 1);
  EXPECT_FALSE(ranges[0].hasLowerBound());
  EXPECT_TRUE(ranges[0].hasUpperBound());
  EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, arrayWithLowerAndUpperBound) {
  auto str = R"(x.y[1:3])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x.y", 1);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 1);
  EXPECT_TRUE(ranges[0].hasLowerBound());
  EXPECT_EQ(ranges[0].getLowerBound(), 1);
  EXPECT_TRUE(ranges[0].hasUpperBound());
  EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, multipleVariables) {
  auto str = R"(x;y[1:3])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto xFilters = vf->getVariableInfo("x", 0);
  ASSERT_EQ(xFilters.size(), 1);
  EXPECT_TRUE(xFilters[0].isVisible());

  auto yFilters = vf->getVariableInfo("y", 1);
  ASSERT_EQ(yFilters.size(), 1);
  EXPECT_TRUE(yFilters[0].isVisible());
}

TEST(VariableFilter, sameVariableRepeated) {
  auto str = R"(x[1:3];x[5:6])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x", 1);
  ASSERT_EQ(filters.size(), 2);

  EXPECT_TRUE(filters[0].isVisible());
  auto ranges0 = filters[0].getRanges();
  EXPECT_EQ(ranges0.size(), 1);
  EXPECT_TRUE(ranges0[0].hasLowerBound());
  EXPECT_EQ(ranges0[0].getLowerBound(), 1);
  EXPECT_TRUE(ranges0[0].hasUpperBound());
  EXPECT_EQ(ranges0[0].getUpperBound(), 3);

  EXPECT_TRUE(filters[1].isVisible());
  auto ranges1 = filters[1].getRanges();
  EXPECT_EQ(ranges0.size(), 1);
  EXPECT_TRUE(ranges1[0].hasLowerBound());
  EXPECT_EQ(ranges1[0].getLowerBound(), 5);
  EXPECT_TRUE(ranges1[0].hasUpperBound());
  EXPECT_EQ(ranges1[0].getUpperBound(), 6);
}

TEST(VariableFilter, lowerRankRequested) {
  auto str = R"(x[$:3,2:7])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x", 1);
  ASSERT_EQ(filters.size(), 1);
  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 1);
  EXPECT_FALSE(ranges[0].hasLowerBound());
  EXPECT_TRUE(ranges[0].hasUpperBound());
  EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, higherRankRequested) {
  auto str = R"(x[$:3,2:7])";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto filters = vf->getVariableInfo("x", 3);
  ASSERT_EQ(filters.size(), 1);

  EXPECT_TRUE(filters[0].isVisible());

  auto ranges = filters[0].getRanges();
  EXPECT_EQ(ranges.size(), 3);

  EXPECT_FALSE(ranges[0].hasLowerBound());
  EXPECT_TRUE(ranges[0].hasUpperBound());
  EXPECT_EQ(ranges[0].getUpperBound(), 3);

  EXPECT_TRUE(ranges[1].hasLowerBound());
  EXPECT_EQ(ranges[1].getLowerBound(), 2);
  EXPECT_TRUE(ranges[1].hasUpperBound());
  EXPECT_EQ(ranges[1].getUpperBound(), 7);

  EXPECT_FALSE(ranges[2].hasLowerBound());
  EXPECT_FALSE(ranges[2].hasUpperBound());
}

TEST(VariableFilter, derivative) {
  auto str = R"(der(x))";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto varFilters = vf->getVariableInfo("x", 0);
  ASSERT_EQ(varFilters.size(), 1);
  EXPECT_FALSE(varFilters[0].isVisible());

  auto derFilters = vf->getVariableDerInfo("x", 0);
  ASSERT_EQ(derFilters.size(), 1);
  EXPECT_TRUE(derFilters[0].isVisible());
}

TEST(VariableFilter, regex) {
  auto str = R"(/^[a-z]+$/)";

  auto vf = VariableFilter::fromString(str);
  ASSERT_TRUE(vf.has_value());
  EXPECT_TRUE(vf->isEnabled());

  auto xFilters = vf->getVariableInfo("x", 0);
  ASSERT_EQ(xFilters.size(), 1);
  EXPECT_TRUE(xFilters[0].isVisible());

  auto yFilters = vf->getVariableInfo("y", 0);
  ASSERT_EQ(yFilters.size(), 1);
  EXPECT_TRUE(yFilters[0].isVisible());

  auto xUnderscoreFilters = vf->getVariableInfo("x_", 0);
  ASSERT_EQ(xUnderscoreFilters.size(), 1);
  EXPECT_FALSE(xUnderscoreFilters[0].isVisible());
}
