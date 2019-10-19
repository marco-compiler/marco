#include "gtest/gtest.h"

#include "forSimulation.hpp"

TEST(ForSimulation, outputTest)	 // NOLINT
{
	runSimulation();
	for (int a = 0; a < 2; a++)
		for (int b = 0; b < 2; b++)
			EXPECT_EQ(IntModifiable[a][b], 3);
}
