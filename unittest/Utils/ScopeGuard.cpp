#include "gtest/gtest.h"
#include "marco/Utils/ScopeGuard.h"

using namespace marco;

TEST(ScopeGuardTest, scopeGuardShouldRestoreInteger)
{
	int val = 4;
	{
		auto g = makeGuard([&]() { val++; });
	}

	EXPECT_EQ(val, 5);
}
