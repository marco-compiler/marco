#include "marco/Runtime/Multithreading/ThreadPool.h"
#include "marco/Runtime/Multithreading/Options.h"

namespace marco::runtime
{
  ThreadPool::ThreadPool()
      : numOfThreads(multithreading::multithreadingOptions().numOfThreads),
        activeThreads(0)
  {
    if (numOfThreads == 0) {
      numOfThreads = 1;
    }

    if (!multithreading::multithreadingOptions().enableMultithreading) {
      numOfThreads = 1;
    }

    for (unsigned int i = 0; i < numOfThreads; ++i) {
      threads.push_back(std::thread(&ThreadPool::threadLoop, this));
    }
  }

  ThreadPool::~ThreadPool()
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      shouldTerminate = true;
    }

    queueCondition.notify_all();

    for (std::thread& activeThread : threads) {
      activeThread.join();
    }

    threads.clear();

    completionCondition.notify_all();
  }

  unsigned int ThreadPool::getNumOfThreads() const
  {
    return numOfThreads;
  }

  void ThreadPool::threadLoop()
  {
    while (true) {
      std::function<void()> job;

      {
        std::unique_lock<std::mutex> lock(queueMutex);

        queueCondition.wait(lock, [this] {
          return !jobs.empty() || shouldTerminate;
        });

        if (shouldTerminate) {
          return;
        }

        job = jobs.front();
        jobs.pop();

        ++activeThreads;
      }

      job();

      {
        std::lock_guard<std::mutex> lock(queueMutex);
        --activeThreads;

        if (activeThreads == 0) {
          completionCondition.notify_all();
        }
      }
    }
  }

  void ThreadPool::async(const std::function<void()>& job)
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      jobs.push(job);
    }

    queueCondition.notify_one();
  }

  void ThreadPool::wait()
  {
    std::unique_lock<std::mutex> lockGuard(queueMutex);

    completionCondition.wait(
        lockGuard, [&] {
          return activeThreads == 0 && jobs.empty();
        });
  }
}
