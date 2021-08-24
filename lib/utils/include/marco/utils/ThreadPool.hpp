#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mutex>
#include <thread>

#include "marco/utils/IRange.hpp"

namespace marco
{
	class ThreadPool;
	class Thread
	{
		public:
		Thread(ThreadPool& tPool): pool(tPool), th([this]() { this->run(); }) {}
		Thread(Thread&& other) = delete;
		Thread(const Thread& other) = delete;
		Thread& operator=(Thread&& other) = delete;
		Thread& operator=(const Thread& other) = delete;
		~Thread() = default;

		void join() { th.join(); }

		private:
		std::function<void()> getTask();
		void run();
		ThreadPool& pool;
		std::thread th;
	};

	class ThreadPool
	{
		public:
		explicit ThreadPool(
				size_t threadCount = std::thread::hardware_concurrency())
				: creatorId(std::this_thread::get_id()), done(false)
		{
			for (auto i : irange(threadCount))
				threads.emplace_back(std::make_unique<Thread>(*this));
		}

		ThreadPool(ThreadPool& other) = delete;
		ThreadPool& operator=(ThreadPool& other) = delete;
		ThreadPool(ThreadPool&& other) = delete;
		ThreadPool& operator=(ThreadPool&& other) = delete;
		~ThreadPool()
		{
			waitUntilQueueEmpty();
			done.store(true, std::memory_order_relaxed);
			for (auto& t : threads)
				addTask([]() {});
			for (auto& t : threads)
				t->join();
		}
		void waitUntilQueueEmpty()
		{
			std::unique_lock<std::mutex> guard(lock);
			queueEmpty.wait(guard, [this]() { return jobs.empty(); });
		}
		std::mutex& getMutex() { return lock; }
		void addTask(std::function<void()> task)
		{
			assert(
					creatorId == std::this_thread::get_id() &&
					"only original thread can push task");
			std::unique_lock<std::mutex> guard(lock);
			jobs.push_back(std::move(task));
			waitingVar.notify_one();
		}
		std::function<void()> getTask()
		{
			std::unique_lock<std::mutex> guard(lock);
			waitingVar.wait(guard, [this]() { return !jobs.empty(); });
			auto job = jobs.front();
			jobs.pop_front();
			queueEmpty.notify_all();
			return job;
		}

		[[nodiscard]] bool empty() { return size() == 0; }

		[[nodiscard]] size_t size()
		{
			std::unique_lock<std::mutex> guard(lock);
			return jobs.size();
		}

		[[nodiscard]] bool isDone() const
		{
			return done.load(std::memory_order_relaxed);
		}

		private:
		std::thread::id creatorId;
		std::mutex lock;
		std::atomic<bool> done;
		std::condition_variable waitingVar;
		std::condition_variable queueEmpty;

		llvm::SmallVector<std::unique_ptr<Thread>, 4> threads;
		std::list<std::function<void()>> jobs;
	};
}	 // namespace marco
