#include "gtest/gtest.h"

#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModCall.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"

using namespace modelica;
using namespace std;

TEST(ModConstTest, constantVectorCanBeAdded)
{
	ModConst l(3, 4, 5);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::sum(l, r), ModConst(8, 8, 10));
}

TEST(ModConstTest, constantVectorAddedAreCasted)
{
	ModConst l(3.0f, 4.0f, 5.0f);
	ModConst r(5, 4, 5);

	auto sum = ModConst::sum(l, r);
	EXPECT_TRUE(sum.isA<float>());

	EXPECT_NEAR(sum.get<float>(0), 8.0f, 0.1);
}
