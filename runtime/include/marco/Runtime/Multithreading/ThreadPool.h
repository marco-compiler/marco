#ifndef MARCO_RUNTIME_MULTITHREADING_THREADPOOL_H
#define MARCO_RUNTIME_MULTITHREADING_THREADPOOL_H

#ifdef THREADS_ENABLE

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace marco::runtime
{
  class ThreadPool
  {
    public:
      ThreadPool();

      ~ThreadPool();

      unsigned int getNumOfThreads() const;

      void async(const std::function<void()>& job);

      void wait();

    private:
      void threadLoop();

    private:
      unsigned int numOfThreads;

      // Tells the threads to stop looking for jobs.
      bool shouldTerminate = false;

      // Prevents data races to the job queue.
      std::mutex queueMutex;

      // Allows threads to wait on new jobs or termination.
      std::condition_variable queueCondition;

      // The number of busy threads.
      unsigned int activeThreads;

      // Signals the completion of all the jobs.
      std::condition_variable completionCondition;

      std::vector<std::thread> threads;
      std::queue<std::function<void()>> jobs;
  };
}

#else

#include <functional>
#include <vector>

namespace marco::runtime
{
  class ThreadPool
  {
    public:
      unsigned int getNumOfThreads() const;

      void async(const std::function<void()>& job);

      void wait();
  };
}

#endif // THREADS_ENABLE

#endif // MARCO_RUNTIME_MULTITHREADING_THREADPOOL_H
