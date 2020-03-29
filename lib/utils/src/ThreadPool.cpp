#include "modelica/utils/ThreadPool.hpp"

#include <functional>

using namespace std;
using namespace modelica;

void Thread::run()
{
	while (!pool->isDone())
		pool->getTask()();
}
