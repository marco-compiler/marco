#include "gtest/gtest.h"

#include "callSimulation.hpp"

extern "C"
{
	void mult(
			int* retVal,
			size_t* retValDims,
			int* base,
			size_t* baseDims,
			int* exp,
			size_t* expDims)
	{
		retVal[0] = base[0] * exp[0];
		EXPECT_EQ(retValDims[0], 1);
		EXPECT_EQ(retValDims[1], 0);
		EXPECT_EQ(baseDims[0], 1);
		EXPECT_EQ(baseDims[1], 0);
		EXPECT_EQ(expDims[0], 1);
		EXPECT_EQ(expDims[1], 0);
	}
}

TEST(CallSimulationTest, powerOfTest)
{
	runSimulation();
	EXPECT_EQ(Int[0], 6);
}
