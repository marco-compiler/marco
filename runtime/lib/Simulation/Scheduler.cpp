#include "marco/Runtime/Simulation/Scheduler.h"
#include "marco/Runtime/Multithreading/ThreadPool.h"
#include "marco/Runtime/Simulation/Options.h"
#include <atomic>
#include <cassert>

#include <iostream>

using namespace ::marco::runtime;

// The thread pool is shared by all the schedulers.
// Having multiple ones would waste resources in instantiating new thread
// groups which would anyway be used one at a time.
static ThreadPool& getSchedulersThreadPool()
{
  static ThreadPool instance;
  return instance;
}

namespace marco::runtime
{
  Scheduler::Equation::Equation(
      EquationFunction function,
      MultidimensionalRange indices,
      bool independentIndices)
      : function(function),
        indices(std::move(indices)),
        independentIndices(independentIndices)
  {
  }

  void Scheduler::addEquation(
      EquationFunction function,
      uint64_t rank,
      int64_t* ranges,
      bool independentIndices)
  {
    std::vector<Range> indices;

    for (uint64_t dim = 0; dim < rank; ++dim) {
      indices.emplace_back(ranges[dim * 2], ranges[dim * 2 + 1]);
    }

    equations.emplace_back(function, std::move(indices), independentIndices);
  }

  void Scheduler::run()
  {
    if (!initialized) {
      initialize();
    }

    ThreadPool& threadPool = getSchedulersThreadPool();
    unsigned int numOfThreads = threadPool.getNumOfThreads();
    std::atomic_size_t chunkIndex = 0;

    for (unsigned int thread = 0; thread < numOfThreads; ++thread) {
      threadPool.async([&]() {
        size_t assignedChunk;

        while ((assignedChunk = chunkIndex++) < threadEquationsChunks.size()) {
          const ThreadEquationsChunk& chunk =
              threadEquationsChunks[assignedChunk];

          const Equation& equation = chunk.first;
          const auto& ranges = chunk.second;
          equation.function(ranges.data());
        }
      });
    }

    threadPool.wait();
  }

  void Scheduler::initialize()
  {
    assert(!initialized && "Scheduler already initialized");

    ThreadPool& threadPool = getSchedulersThreadPool();
    unsigned int numOfThreads = threadPool.getNumOfThreads();

    int64_t chunksFactor = simulation::getOptions().threadEquationChunks;
    int64_t minNumOfEquations = numOfThreads * chunksFactor;

    for (const Equation& equation : equations) {
      bool shouldSplitIndices = equation.independentIndices;
      uint64_t flatSize = getFlatSize(equation.indices);

      if (shouldSplitIndices) {
        // Avoid splitting the indices until a certain amount of equations.
        shouldSplitIndices &=
            static_cast<int64_t>(flatSize) >= minNumOfEquations;
      }

      if (shouldSplitIndices) {
        uint64_t chunkSize =
            std::max(flatSize / numOfThreads, static_cast<uint64_t>(1));

        uint64_t equationFlatIndex = 0;

        // Divide the ranges into chunks.
        while (equationFlatIndex < flatSize) {
          uint64_t beginFlatIndex = equationFlatIndex;

          uint64_t endFlatIndex = std::min(
              beginFlatIndex + static_cast<uint64_t>(chunkSize), flatSize);

          std::vector<int64_t> beginIndices;
          std::vector<int64_t> endIndices;

          getIndicesFromFlatIndex(
              beginFlatIndex, beginIndices, equation.indices);

          if (endFlatIndex == flatSize) {
            endIndices.clear();

            for (const Range& range : equation.indices) {
              endIndices.push_back(range.end);
            }
          } else {
            getIndicesFromFlatIndex(
                endFlatIndex, endIndices, equation.indices);
          }

          assert(beginIndices.size() == endIndices.size());
          std::vector<int64_t> ranges;

          for (size_t i = 0, e = beginIndices.size(); i < e; ++i) {
            ranges.push_back(beginIndices[i]);
            ranges.push_back(endIndices[i]);
          }

          threadEquationsChunks.emplace_back(equation, std::move(ranges));

          // Move to the next chunk.
          equationFlatIndex = endFlatIndex;
        }
      } else {
        std::vector<int64_t> ranges;

        for (const Range& range : equation.indices) {
          ranges.push_back(range.begin);
          ranges.push_back(range.end);
        }

        threadEquationsChunks.emplace_back(equation, std::move(ranges));
      }
    }

    initialized = true;
  }
}

//===---------------------------------------------------------------------===//
// schedulerCreate

[[maybe_unused]] static void* schedulerCreate_pvoid()
{
  auto* instance = new Scheduler();
  return static_cast<void*>(instance);
}

RUNTIME_FUNC_DEF(schedulerCreate, PTR(void))

//===---------------------------------------------------------------------===//
// schedulerDestroy

[[maybe_unused]] static void schedulerDestroy_void(void* scheduler)
{
  assert(scheduler != nullptr);
  delete static_cast<Scheduler*>(scheduler);
}

RUNTIME_FUNC_DEF(schedulerDestroy, void, PTR(void))

//===---------------------------------------------------------------------===//
// schedulerAddEquation

[[maybe_unused]] static void schedulerAddEquation_void(
    void* scheduler,
    void* equationFunction,
    uint64_t rank,
    int64_t* ranges,
    bool independentIndices)
{
  assert(scheduler != nullptr);

  static_cast<Scheduler*>(scheduler)->addEquation(
      reinterpret_cast<Scheduler::EquationFunction>(equationFunction),
      rank, ranges, independentIndices);
}

RUNTIME_FUNC_DEF(schedulerAddEquation, void, PTR(void), PTR(void), uint64_t, PTR(int64_t), bool)

//===---------------------------------------------------------------------===//
// schedulerRun

[[maybe_unused]] static void schedulerRun_void(void* scheduler)
{
  assert(scheduler != nullptr);
  static_cast<Scheduler*>(scheduler)->run();
}

RUNTIME_FUNC_DEF(schedulerRun, void, PTR(void))
