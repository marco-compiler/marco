#include "modelica/utils/ScopeGuard.hpp"

#include "gtest/gtest.h"

using namespace modelica;

TEST(ScopeGuardTest, scopeGuardShouldRestoreInteger)
{
	int val = 4;
	{
		auto g = makeGuard([&]() { val++; });
	}

	EXPECT_EQ(val, 5);
}
