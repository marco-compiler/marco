#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mutex>
#include <thread>

#include "modelica/utils/IRange.hpp"

namespace modelica
{
	class ThreadPool;
	class Thread
	{
		public:
		Thread(ThreadPool& tPool)
				: pool(&tPool),
					th(std::make_unique<std::thread>([this]() { this->run(); }))
		{
		}
		Thread(Thread&& other): pool(other.pool), th(std::move(other.th)) {}

		Thread(const Thread& other) = delete;
		Thread& operator=(Thread&& other) = delete;
		Thread& operator=(const Thread& other) = delete;
		~Thread() = default;

		void join() { th->join(); }

		private:
		std::function<void()> getTask();
		void run();
		ThreadPool* pool;
		std::unique_ptr<std::thread> th;
	};

	class ThreadPool
	{
		public:
		explicit ThreadPool(
				size_t threadCount = std::thread::hardware_concurrency())
				: creatorId(std::this_thread::get_id()), threads(), done(false)
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
			done.store(true, std::memory_order_relaxed);
			for (auto& t : threads)
				addTask([]() {});
			waitingVar.notify_all();
			for (auto& t : threads)
				t->join();
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
			return jobs.pop_back_val();
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
		llvm::SmallVector<std::unique_ptr<Thread>, 4> threads;
		llvm::SmallVector<std::function<void()>, 0> jobs;
		std::mutex lock;
		std::atomic<bool> done;
		std::condition_variable waitingVar;
	};
}	 // namespace modelica
