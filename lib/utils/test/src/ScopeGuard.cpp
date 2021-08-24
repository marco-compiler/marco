#include "marco/utils/ScopeGuard.hpp"

#include "gtest/gtest.h"

using namespace marco;

TEST(ScopeGuardTest, scopeGuardShouldRestoreInteger)
{
	int val = 4;
	{
		auto g = makeGuard([&]() { val++; });
	}

	EXPECT_EQ(val, 5);
}
