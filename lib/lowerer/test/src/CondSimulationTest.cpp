#include "gtest/gtest.h"

#include "condSimulation.hpp"

TEST(CondSimulation, outputTest)	// NOLINT
{
	runSimulation();
	EXPECT_EQ(res[0], 9);
}
