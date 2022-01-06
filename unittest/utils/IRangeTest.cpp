#include "gtest/gtest.h"
#include <algorithm>

#include "marco/utils/IRange.hpp"

using namespace marco;
using namespace llvm;

TEST(IRangeTest, irangeCanBeUsedInRangeFor)
{
	int a = 0;
	for (auto num : irange(0, 5))
		EXPECT_EQ(num, a++);
	EXPECT_EQ(a, 5);
}

TEST(IRangeTest, irangeCanBeUnsigned)
{
	unsigned a = 0;
	for (auto num : irange<unsigned>(0, 5))
		EXPECT_EQ(num, a++);

	EXPECT_EQ(a, 5);
}

TEST(IRangeTest, irangeWithSingleArgument)
{
	int a = 0;
	for (auto num : irange(5))
		EXPECT_EQ(num, a++);
	EXPECT_EQ(a, 5);
}
