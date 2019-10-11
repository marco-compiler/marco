#include "gtest/gtest.h"

#include "floatSimulation.hpp"

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
