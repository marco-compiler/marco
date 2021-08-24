#include "marco/utils/ThreadPool.hpp"

#include <functional>

using namespace std;
using namespace marco;

void Thread::run()
{
	while (!pool.isDone())
		pool.getTask()();
}
