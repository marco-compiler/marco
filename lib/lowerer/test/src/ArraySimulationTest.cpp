#include "gtest/gtest.h"

#include "arraySimulation.hpp"

TEST(ArraySimulationTest, intSumArrayTest)
{
	runSimulation();
	for (int a = 0; a < 3; a++)
		EXPECT_EQ(30 + a + 1, intVector[a][0]);
}

TEST(ArraySimulationTest, intConstantArrayTest)
{
	runSimulation();
	for (int a = 0; a < 3; a++)
		EXPECT_EQ(-(a + 1), intVectorConstant[a][0]);
}
