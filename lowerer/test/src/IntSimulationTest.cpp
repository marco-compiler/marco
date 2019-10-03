#include "gtest/gtest.h"

extern "C"
{
	extern int X;
	void runSimulation();
}

TEST(IntSimulationTest, defaultTest)	// NOLINT
{
	EXPECT_EQ(X, 0);
	runSimulation();
	EXPECT_EQ(X, 9);
}
