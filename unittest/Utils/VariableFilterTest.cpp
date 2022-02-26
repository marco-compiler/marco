#include "gtest/gtest.h"
#include "marco/Utils/VariableFilter.h"

using namespace marco;

TEST(VariableFilter, scalarVariable)
{
	std::string str = "x";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 0);
	EXPECT_TRUE(x.isVisible());
	EXPECT_TRUE(x.getRanges().empty());
}

TEST(VariableFilter, unboundedArray)
{
	std::string str = "x[$:$]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 1);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_FALSE(ranges[0].hasLowerBound());
	EXPECT_FALSE(ranges[0].hasUpperBound());
}

TEST(VariableFilter, arrayWithLowerBound)
{
	std::string str = "x[1:$]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 1);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_TRUE(ranges[0].hasLowerBound());
	EXPECT_EQ(ranges[0].getLowerBound(), 1);
	EXPECT_FALSE(ranges[0].hasUpperBound());
}

TEST(VariableFilter, arrayWithUpperBound)
{
	std::string str = "x[$:3]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 1);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_FALSE(ranges[0].hasLowerBound());
	EXPECT_TRUE(ranges[0].hasUpperBound());
	EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, arrayWithLowerAndUpperBound)
{
	std::string str = "x[1:3]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 1);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_TRUE(ranges[0].hasLowerBound());
	EXPECT_EQ(ranges[0].getLowerBound(), 1);
	EXPECT_TRUE(ranges[0].hasUpperBound());
	EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, multipleVariables)
{
	std::string str = "x;y[1:3]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 0);
	EXPECT_TRUE(x.isVisible());

	auto y = vf->getVariableInfo("y", 0);
	EXPECT_TRUE(y.isVisible());
}

TEST(VariableFilter, lowerRankRequested)
{
	std::string str = "x[$:3,2:7]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 1);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_FALSE(ranges[0].hasLowerBound());
	EXPECT_TRUE(ranges[0].hasUpperBound());
	EXPECT_EQ(ranges[0].getUpperBound(), 3);
}

TEST(VariableFilter, higherRankRequested)
{
	std::string str = "x[$:3,2:7]";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 3);
	EXPECT_TRUE(x.isVisible());

	auto ranges = x.getRanges();
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

TEST(VariableFilter, derivative)
{
	std::string str = "der(x)";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 0);
	EXPECT_FALSE(x.isVisible());

	auto der_x = vf->getVariableDerInfo("x", 0);
	EXPECT_TRUE(der_x.isVisible());
}

TEST(VariableFilter, regex)
{
	std::string str = "/^[a-z]+$/";
	auto vf = VariableFilter::fromString(str);
	ASSERT_FALSE(!vf);

	EXPECT_TRUE(vf->isEnabled());

	auto x = vf->getVariableInfo("x", 0);
	EXPECT_TRUE(x.isVisible());

	auto y = vf->getVariableInfo("y", 0);
	EXPECT_TRUE(y.isVisible());

	auto x_ = vf->getVariableInfo("x_", 0);
	EXPECT_FALSE(x_.isVisible());
}
