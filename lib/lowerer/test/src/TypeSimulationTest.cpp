#include "gtest/gtest.h"

#include "typeSimulation.hpp"

TEST(IntSimulationTest, intOutPutTest)	// NOLINT
{
	runSimulation();
	EXPECT_EQ(IntModifiable[0], 9);
	EXPECT_EQ(IntConstant[0], 6);
}

TEST(IntSimulationTest, boolOutPutTest)	 // NOLINT
{
	runSimulation();
	EXPECT_EQ(BoolConstant[0], false);
}
