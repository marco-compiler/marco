#include "gtest/gtest.h"
#include <vector>

#include "modelica/utils/IRange.hpp"
#include "modelica/utils/ThreadPool.hpp"

using namespace modelica;
using namespace std;

TEST(ThreadPoolTest, testCreation)
{
	vector<int> vect(300, 0);
	{
		ThreadPool pool;
		for (size_t i : irange(vect.size()))
			pool.addTask([&vect, i]() { vect[i] = 1; });

		while (!pool.empty())
			;
	}

	EXPECT_TRUE(find(vect.begin(), vect.end(), 0) == vect.end());
}
