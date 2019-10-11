#include "gtest/gtest.h"

#include "typeSimulation.hpp"

TEST(IntSimulationTest, intOutPutTest)	// NOLINT
{
	runSimulation();
	EXPECT_EQ(IntModifiable[0], 9);
	EXPECT_EQ(IntConstant[0], 6);
}

TEST(IntSimulationTest, boolOutPutTest)	// NOLINT
{
	runSimulation();
	EXPECT_EQ(BoolConstant[0], false);
}

TEST(IntSimulationTest, floatOutputTest)	// NOLINT
{
	runSimulation();
	EXPECT_NEAR(FloatConstant[0], 3, 0.1F);
}

TEST(IntSimulationTest, floatModifiableOutPutTest)	// NOLINT
{
	runSimulation();
	EXPECT_NEAR(FloatModifiable[0], 13, 0.1F);
}
