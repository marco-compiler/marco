#include "marco/Runtime/Simulation/Scheduler.h"
#include "marco/Runtime/Multithreading/ThreadPool.h"
#include "marco/Runtime/Simulation/Options.h"
#include <atomic>
#include <cassert>
#include <iostream>
#include <optional>

using namespace ::marco::runtime;

//===---------------------------------------------------------------------===//
// Profiling
//===---------------------------------------------------------------------===//

#ifdef MARCO_PROFILING
#include "marco/Runtime/Simulation/Profiler.h"

SchedulerProfiler::SchedulerProfiler(int64_t schedulerId)
    : Profiler("Scheduler " + std::to_string(schedulerId))
{
}

void SchedulerProfiler::createChunksGroupCounters(size_t amount)
{
  chunksGroupsCounters.clear();
  chunksGroupsCounters.resize(amount, 0);
}

void SchedulerProfiler::createChunksGroupTimers(size_t amount)
{
  chunksGroups.clear();

  for (size_t i = 0; i < amount; ++i) {
    chunksGroups.push_back(std::make_unique<profiling::Timer>());
  }
}

void SchedulerProfiler::reset()
{
  std::lock_guard<std::mutex> lockGuard(mutex);

  addEquation.reset();
  initialization.reset();
  run.reset();
  sequentialRuns = 0;
  multithreadedRuns = 0;

  for (auto& chunksGroup : chunksGroups) {
    chunksGroup->reset();
  }
}

void SchedulerProfiler::print() const
{
  std::lock_guard<std::mutex> lockGuard(mutex);

  std::cerr << "Time spent on adding the equations: "
            << addEquation.totalElapsedTime<std::milli>() << " ms"
            << std::endl;

  std::cerr << "Time spent on initialization: "
            << initialization.totalElapsedTime<std::milli>() << " ms"
            << std::endl;

  std::cerr << "Time spent on 'run' method: "
            << run.totalElapsedTime<std::milli>()<< " ms" << std::endl;

  std::cerr << "Number of sequential executions: " << sequentialRuns
            << std::endl;

  std::cerr << "Number of multithreaded executions: " << multithreadedRuns
            << std::endl;

  for (size_t i = 0, e = chunksGroups.size(); i < e; ++i) {
    auto chunksGroupsCounter = chunksGroupsCounters[i];

    double averageChunksGroupTime =
        chunksGroups[i]->totalElapsedTime<std::nano>() /
            static_cast<double>(chunksGroupsCounter);

    std::cerr << "\n";

    std::cerr << "Time (total) spent by thread #" << i
              << " in processing equations: "
              << chunksGroups[i]->totalElapsedTime<std::milli>() << " ms"
              << std::endl;

    std::cerr << "Time (average) spent by thread #" << i
              << " in processing equations: " << averageChunksGroupTime
              << " ns" << std::endl;

    std::cerr << "Number of chunks groups processed: "
              << chunksGroupsCounter << std::endl;
  }
}

#define SCHEDULER_PROFILER_ADD_EQUATION_START profiler->addEquation.start()
#define SCHEDULER_PROFILER_ADD_EQUATION_STOP profiler->addEquation.stop()

#define SCHEDULER_PROFILER_RUN_START profiler->run.start()
#define SCHEDULER_PROFILER_RUN_STOP profiler->run.stop()

#define SCHEDULER_PROFILER_INCREMENT_SEQUENTIAL_RUNS_COUNTER ++profiler->sequentialRuns
#define SCHEDULER_PROFILER_INCREMENT_MULTITHREADED_RUNS_COUNTER ++profiler->multithreadedRuns

#define SCHEDULER_PROFILER_INITIALIZATION_START profiler->initialization.start()
#define SCHEDULER_PROFILER_INITIALIZATION_STOP profiler->initialization.stop()

#define SCHEDULER_PROFILER_CHUNKS_GROUP_START(thread) \
    profiler->chunksGroups[thread]->start();          \
    profiler->chunksGroupsCounters[thread]++

#define SCHEDULER_PROFILER_CHUNKS_GROUP_STOP(thread) profiler->chunksGroups[thread]->stop()

#else

#define SCHEDULER_PROFILER_DO_NOTHING static_assert(true)

#define SCHEDULER_PROFILER_ADD_EQUATION_START SCHEDULER_PROFILER_DO_NOTHING
#define SCHEDULER_PROFILER_ADD_EQUATION_STOP SCHEDULER_PROFILER_DO_NOTHING

#define SCHEDULER_PROFILER_RUN_START SCHEDULER_PROFILER_DO_NOTHING
#define SCHEDULER_PROFILER_RUN_STOP SCHEDULER_PROFILER_DO_NOTHING

#define SCHEDULER_PROFILER_INCREMENT_SEQUENTIAL_RUNS_COUNTER SCHEDULER_PROFILER_DO_NOTHING
#define SCHEDULER_PROFILER_INCREMENT_MULTITHREADED_RUNS_COUNTER SCHEDULER_PROFILER_DO_NOTHING

#define SCHEDULER_PROFILER_INITIALIZATION_START SCHEDULER_PROFILER_DO_NOTHING
#define SCHEDULER_PROFILER_INITIALIZATION_STOP SCHEDULER_PROFILER_DO_NOTHING

#define SCHEDULER_PROFILER_CHUNKS_GROUP_START(thread) SCHEDULER_PROFILER_DO_NOTHING
#define SCHEDULER_PROFILER_CHUNKS_GROUP_STOP(thread) SCHEDULER_PROFILER_DO_NOTHING

#endif

//===---------------------------------------------------------------------===//
// Scheduler
//===---------------------------------------------------------------------===//

static int64_t getUniqueSchedulerIdentifier()
{
  static int64_t identifier = 0;
  return identifier++;
}

// The thread pool is shared by all the schedulers.
// Having multiple ones would waste resources in instantiating new thread
// groups which would anyway be used one at a time.
static ThreadPool& getSchedulersThreadPool()
{
  static ThreadPool instance;
  return instance;
}

static uint64_t getEquationsChunkSize(
    const Scheduler::ThreadEquationsChunk& chunk)
{
  int64_t result = 1;

  assert(chunk.second.size() % 2 == 0);
  size_t rank = chunk.second.size() / 2;

  for (size_t dim = 0; dim < rank; ++dim) {
    auto lowerBound = chunk.second[dim * 2];
    auto upperBound = chunk.second[dim * 2 + 1];
    auto size = upperBound - lowerBound;
    result *= size;
  }

  return result;
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

  Scheduler::Scheduler()
  {
    identifier = getUniqueSchedulerIdentifier();

#ifdef MARCO_PROFILING
    ThreadPool& threadPool = getSchedulersThreadPool();
    unsigned int numOfThreads = threadPool.getNumOfThreads();

    profiler = std::make_shared<SchedulerProfiler>(identifier);
    profiler->createChunksGroupCounters(numOfThreads);
    profiler->createChunksGroupTimers(numOfThreads);

    registerProfiler(profiler);
#endif
  }

  void Scheduler::addEquation(
      EquationFunction function,
      uint64_t rank,
      int64_t* ranges,
      bool independentIndices)
  {
    SCHEDULER_PROFILER_ADD_EQUATION_START;
    std::vector<Range> indices;

    for (uint64_t dim = 0; dim < rank; ++dim) {
      indices.emplace_back(ranges[dim * 2], ranges[dim * 2 + 1]);
    }

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Scheduler " << identifier << "] New equation added"
                << std::endl;

      std::cerr << "  - Rank: " << rank << std::endl;

      if (rank != 0) {
        std::cerr << "  - Ranges: ";

        for (uint64_t i = 0; i < rank; ++i) {
          std::cerr << "[" << indices[i].begin << ", " << indices[i].end << ")";
        }

        std::cerr << std::endl;
      }
    }

    equations.emplace_back(function, std::move(indices), independentIndices);
    SCHEDULER_PROFILER_ADD_EQUATION_STOP;
  }

  void Scheduler::initialize()
  {
    SCHEDULER_PROFILER_INITIALIZATION_START;
    assert(!initialized && "Scheduler already initialized");

    ThreadPool& threadPool = getSchedulersThreadPool();
    unsigned int numOfThreads = threadPool.getNumOfThreads();
    int64_t chunksFactor = simulation::getOptions().equationsChunksFactor;
    int64_t numOfChunks = numOfThreads * chunksFactor;

    uint64_t numOfScalarEquations = 0;

    for (const Equation& equation : equations) {
      numOfScalarEquations += getFlatSize(equation.indices);
    }

    size_t chunksGroupMaxSize =
        (numOfScalarEquations + numOfChunks - 1) / numOfChunks;

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Scheduler " << identifier << "] Initializing" << std::endl
                << "  - Number of equations: " << numOfScalarEquations
                << std::endl
                << "  - Number of threads: " << numOfThreads << std::endl
                << "  - Chunks factor: " << chunksFactor << std::endl
                << "  - Number of chunks: " << numOfChunks << std::endl
                << "  - Max size of chunks group: " << chunksGroupMaxSize
                << std::endl;
    }

    std::vector<ThreadEquationsChunk> chunksGroup;
    size_t chunksGroupSize = 0;

    auto pushChunksGroupFn = [&]() {
      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "[Scheduler " << identifier << "] Adding chunks group" << std::endl;
        std::cerr << "  - Number of chunks: " << chunksGroup.size()
                  << std::endl;

        uint64_t totalSize = 0;
        std::cerr << "  - Chunks sizes: [";

        for (size_t i = 0, e = chunksGroup.size(); i < e; ++i) {
          if (i != 0) {
            std::cerr << ", ";
          }

          uint64_t chunkSize = getEquationsChunkSize(chunksGroup[i]);
          std::cerr << chunkSize;
          totalSize += chunkSize;
        }

        std::cerr << "]" << std::endl;

        for (const auto& chunk : chunksGroup) {
          std::cerr << "    - Function: "
                    << reinterpret_cast<void*>(chunk.first.function)
                    << std::endl;

          std::cerr << "    - Range: ";

          assert(chunk.second.size() % 2 == 0);
          size_t rank = chunk.second.size() / 2;

          for (size_t dim = 0; dim < rank; ++dim) {
            auto lowerBound = chunk.second[dim * 2];
            auto upperBound = chunk.second[dim * 2 + 1];
            std::cerr << "[" << lowerBound << ", " << upperBound << ")";
          }

          std::cerr << std::endl;
        }

        std::cerr << "  - Total size: " << totalSize << std::endl;
      }

      threadEquationsChunks.push_back(std::move(chunksGroup));
      chunksGroup.clear();
      chunksGroupSize = 0;
    };

    for (const Equation& equation : equations) {
      uint64_t flatSize = getFlatSize(equation.indices);
      size_t remainingSpace = chunksGroupMaxSize - chunksGroupSize;

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "[Scheduler " << identifier << "] Partitioning equation"
                  << std::endl;

        std::cerr << "  - Function: "
                  << reinterpret_cast<void*>(equation.function) << std::endl;

        std::cerr << "  - Ranges: ";

        for (const auto& range : equation.indices) {
          std::cerr << "[" << range.begin << ", " << range.end << ")";
        }

        std::cerr << std::endl;
        std::cerr << "  - Flat size: " << flatSize << std::endl;

        std::cerr << "  - Independent indices: "
                  << (equation.independentIndices ? "true" : "false")
                  << std::endl;

        std::cerr << "  - Remaining space: " << remainingSpace << std::endl;
      }

      if (equation.independentIndices) {
        uint64_t equationFlatIndex = 0;
        size_t equationRank = equation.indices.size();

        // Divide the ranges into chunks.
        while (equationFlatIndex < flatSize) {
          uint64_t beginFlatIndex = equationFlatIndex;

          uint64_t endFlatIndex = std::min(
              beginFlatIndex + static_cast<uint64_t>(remainingSpace),
              flatSize);

          assert(endFlatIndex > 0);
          --endFlatIndex;

          std::vector<int64_t> beginIndices;
          std::vector<int64_t> endIndices;

          getIndicesFromFlatIndex(
              beginFlatIndex, beginIndices, equation.indices);

          getIndicesFromFlatIndex(
              endFlatIndex, endIndices, equation.indices);

          assert(beginIndices.size() == equationRank);
          assert(endIndices.size() == equationRank);

          if (marco::runtime::simulation::getOptions().debug) {
            std::cerr << "    - Begin indices: [";

            size_t rank = beginIndices.size();

            for (size_t dim = 0; dim < rank; ++dim) {
              if (dim != 0) {
                std::cerr << ", ";
              }

              std::cerr << beginIndices[dim];
            }

             std::cerr << "]" << std::endl;
             std::cerr << "      End indices: [";

             for (size_t dim = 0; dim < rank; ++dim) {
               if (dim != 0) {
                 std::cerr << ", ";
               }

               std::cerr << endIndices[dim];
             }

             std::cerr << "]" << std::endl;
          }

          std::vector<std::vector<int64_t>> unwrappingBeginIndices;
          std::vector<std::vector<int64_t>> unwrappingEndIndices;

          // We need to detect if some of the dimensions do wrap around.
          // In this case, the indices must be split into multiple ranges.
          std::optional<size_t> increasingDimension = std::nullopt;

          for (size_t dim = 0; dim < equationRank; ++dim) {
            if (endIndices[dim] > beginIndices[dim] &&
                dim + 1 != equationRank) {
              increasingDimension = dim;
              break;
            }
          }

          if (increasingDimension) {
            if (marco::runtime::simulation::getOptions().debug) {
              std::cerr << "    - Increasing dimension: "
                        << *increasingDimension << std::endl;
            }

            std::vector<int64_t> currentBeginIndices(beginIndices);
            std::vector<int64_t> currentEndIndices(beginIndices);
            currentEndIndices.back() = equation.indices.back().end - 1;

            unwrappingBeginIndices.push_back(currentBeginIndices);
            unwrappingEndIndices.push_back(currentEndIndices);

            for (size_t i = 0, e = equationRank - *increasingDimension - 2;
                 i < e; ++i) {
              currentBeginIndices[equationRank - i - 1] = 0;

              currentEndIndices[equationRank - i - 2] =
                  equation.indices[equationRank - i - 2].end - 1;

              if (currentBeginIndices[equationRank - i - 2] + 1 !=
                  equation.indices[equationRank - i - 2].end) {
                ++currentBeginIndices[equationRank - i - 2];

                unwrappingBeginIndices.push_back(currentBeginIndices);
                unwrappingEndIndices.push_back(currentEndIndices);
              }
            }

            currentBeginIndices[*increasingDimension + 1] = 0;

            if (endIndices[*increasingDimension] -
                    beginIndices[*increasingDimension] > 1) {
              ++currentBeginIndices[*increasingDimension];

              currentEndIndices[*increasingDimension] =
                  endIndices[*increasingDimension] - 1;

              unwrappingBeginIndices.push_back(currentBeginIndices);
              unwrappingEndIndices.push_back(currentEndIndices);
            }

            for (size_t i = 0, e = equationRank - *increasingDimension - 1;
                 i < e; ++i) {
              currentBeginIndices[*increasingDimension + i] =
                  endIndices[*increasingDimension + i];

              currentEndIndices[*increasingDimension + i] =
                  endIndices[*increasingDimension + i];

              currentEndIndices[*increasingDimension + i + 1] =
                  endIndices[*increasingDimension + i + 1];

              if (currentEndIndices[*increasingDimension + i + 1] != 0) {
                --currentEndIndices[*increasingDimension + i + 1];
                unwrappingBeginIndices.push_back(currentBeginIndices);
                unwrappingEndIndices.push_back(currentEndIndices);
              }
            }

            currentBeginIndices.back() = endIndices.back();
            currentEndIndices.back() = endIndices.back();
            unwrappingBeginIndices.push_back(currentBeginIndices);
            unwrappingEndIndices.push_back(currentEndIndices);
          } else {
            if (marco::runtime::simulation::getOptions().debug) {
              std::cerr << "    - Increasing dimension not found" << std::endl;
            }

            unwrappingBeginIndices.push_back(std::move(beginIndices));
            unwrappingEndIndices.push_back(std::move(endIndices));
          }

          assert(unwrappingBeginIndices.size() == unwrappingEndIndices.size());

          if (marco::runtime::simulation::getOptions().debug) {
            for (size_t unwrappingIndex = 0;
                 unwrappingIndex < unwrappingBeginIndices.size();
                 ++unwrappingIndex) {
              std::cerr << "    - #" << unwrappingIndex
                        << " Unwrapping begin indices: [";

              size_t rank = unwrappingBeginIndices[unwrappingIndex].size();

              for (size_t dim = 0; dim < rank; ++dim) {
                if (dim != 0) {
                  std::cerr << ", ";
                }

                std::cerr << unwrappingBeginIndices[unwrappingIndex][dim];
              }

              std::cerr << "]" << std::endl;
              std::cerr << "      #" << unwrappingIndex
                        <<" Unwrapping end indices:   [";

              for (size_t dim = 0; dim < rank; ++dim) {
                if (dim != 0) {
                  std::cerr << ", ";
                }

                std::cerr << unwrappingEndIndices[unwrappingIndex][dim];
              }

              std::cerr << "]" << std::endl;
            }
          }

          for (size_t i = 0, e = unwrappingBeginIndices.size(); i < e; ++i) {
            std::vector<int64_t> ranges;

            for (size_t j = 0; j < equationRank; ++j) {
              const auto& currentBeginIndices = unwrappingBeginIndices[i];
              const auto& currentEndIndices = unwrappingEndIndices[i];

              assert(currentBeginIndices[j] <= currentEndIndices[j]);
              ranges.push_back(currentBeginIndices[j]);
              ranges.push_back(currentEndIndices[j] + 1);
            }

            chunksGroup.emplace_back(equation, ranges);
          }

          // Move to the next chunks.
          endFlatIndex = getFlatIndex(
              unwrappingEndIndices.back(), equation.indices);

          equationFlatIndex = endFlatIndex + 1;

          // Create a new chunks group if necessary.
          chunksGroupSize += equationFlatIndex - beginFlatIndex;

          if (chunksGroupSize >= chunksGroupMaxSize) {
            pushChunksGroupFn();
          }
        }
      } else {
        // All the indices must be visited by a single thread.
        std::vector<int64_t> ranges;

        for (const Range& range : equation.indices) {
          ranges.push_back(range.begin);
          ranges.push_back(range.end);
        }

        if (flatSize <= remainingSpace) {
          // There is still space in the current chunks group.
          chunksGroup.emplace_back(equation, ranges);
          chunksGroupSize += flatSize;

          if (chunksGroupSize >= chunksGroupMaxSize) {
            pushChunksGroupFn();
          }
        } else {
          if (flatSize >= chunksGroupMaxSize) {
            // Independent chunks group exceeding the maximum number of
            // equations inside a chunk.
            if (marco::runtime::simulation::getOptions().debug) {
              std::cerr << "[Scheduler " << identifier
                        << "] Equation independently exceeds the maximum size "
                           "for a group" << std::endl;
            }

            std::vector<ThreadEquationsChunk> independentChunksGroup;
            independentChunksGroup.emplace_back(equation, ranges);
            threadEquationsChunks.push_back(std::move(independentChunksGroup));
          } else {
            pushChunksGroupFn();
            chunksGroup.emplace_back(equation, ranges);
            chunksGroupSize += flatSize;

            if (chunksGroupSize >= chunksGroupMaxSize) {
              pushChunksGroupFn();
            }
          }
        }
      }
    }

    if (chunksGroupSize != 0) {
      pushChunksGroupFn();
    }

    assert(std::all_of(
               equations.begin(), equations.end(),
               [&](const Equation& equation) {
                 return checkEquationScheduledExactlyOnce(equation);
               }) && "Not all the equations are scheduled exactly once");

    assert(std::all_of(
        threadEquationsChunks.begin(), threadEquationsChunks.end(),
        [&](const std::vector<ThreadEquationsChunk>& group) {
          return std::all_of(
              group.begin(), group.end(),
              [&](const ThreadEquationsChunk& chunk) {
                return checkEquationIndicesExistence(chunk);
              });
        }) && "Some nonexistent equation indices have been scheduled");

    initialized = true;

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Scheduler " << identifier << "] Initialized" << std::endl;

      std::cerr << "  - Number of chunks groups: "
                << threadEquationsChunks.size() << std::endl;
    }

    SCHEDULER_PROFILER_INITIALIZATION_STOP;
  }

  bool Scheduler::checkEquationScheduledExactlyOnce(
      const Equation& equation) const
  {
    auto beginIndicesIt =
        MultidimensionalRangeIterator::begin(equation.indices);

    auto endIndicesIt =
        MultidimensionalRangeIterator::end(equation.indices);

    size_t rank = equation.indices.size();

    for (auto it = beginIndicesIt; it != endIndicesIt; ++it) {
      std::vector<int64_t> indices;

      for (size_t dim = 0; dim < rank; ++dim) {
        indices.push_back((*it)[dim]);
      }

      size_t chunksCount = 0;

      for (const auto& chunksGroup : threadEquationsChunks) {
        chunksCount += std::count_if(
            chunksGroup.begin(), chunksGroup.end(),
            [&](const ThreadEquationsChunk& chunk) {
              if (chunk.first.function != equation.function) {
                return false;
              }

              bool containsPoint = true;

              for (size_t dim = 0; dim < rank && containsPoint; ++dim) {
                if (!(indices[dim] >= chunk.second[dim * 2] &&
                      indices[dim] < chunk.second[dim * 2 + 1])) {
                  containsPoint = false;
                }
              }

              return containsPoint;
            });
      }

      if (chunksCount != 1) {
        return false;
      }
    }

    return true;
  }

  bool Scheduler::checkEquationIndicesExistence(
      const ThreadEquationsChunk& chunk) const
  {
    const auto& equationIndices = chunk.first.indices;
    assert(chunk.second.size() % 2 == 0);
    size_t rank = chunk.second.size() / 2;

    for (size_t dim = 0; dim < rank; ++dim) {
      auto lowerBound = chunk.second[dim * 2];
      auto upperBound = chunk.second[dim * 2 + 1];

      if (lowerBound < equationIndices[dim].begin) {
        return false;
      }

      if (upperBound > equationIndices[dim].end) {
        return false;
      }
    }

    return true;
  }

  void Scheduler::run()
  {
    SCHEDULER_PROFILER_RUN_START;

    if (!initialized) {
      initialize();
    }

    int64_t calibrationRuns =
        marco::runtime::simulation::getOptions().schedulerCalibrationRuns;

    bool isSequentialCalibrationRun = runsCounter < calibrationRuns;

    bool isMultithreadedCalibrationRun =
        runsCounter >= calibrationRuns &&
        runsCounter < calibrationRuns * 2;

    if (isSequentialCalibrationRun) {
      runSequentialWithCalibration();
    } else if (isMultithreadedCalibrationRun) {
      runMultithreadedWithCalibration();

      bool isLastCalibrationRound = runsCounter == (calibrationRuns * 2 - 1);

      if (isLastCalibrationRound) {
        runStrategy = sequentialRunsMinTime < multithreadedRunsMinTime
            ? RunStrategy::Sequential : RunStrategy::Multithreaded;
      }
    } else {
      if (runStrategy == RunStrategy::Sequential) {
        runSequential();
      } else {
        runMultithreaded();
      }
    }

    ++runsCounter;
    SCHEDULER_PROFILER_RUN_STOP;
  }

  void Scheduler::runSequential()
  {
    SCHEDULER_PROFILER_INCREMENT_SEQUENTIAL_RUNS_COUNTER;

    for (const auto& chunksGroup : threadEquationsChunks) {
      SCHEDULER_PROFILER_CHUNKS_GROUP_START(0);

      for (const ThreadEquationsChunk& chunk : chunksGroup) {
        const Equation& equation = chunk.first;
        const auto& ranges = chunk.second;
        equation.function(ranges.data());
      }

      SCHEDULER_PROFILER_CHUNKS_GROUP_STOP(0);
    }
  }

  void Scheduler::runSequentialWithCalibration()
  {
    // Measure the time spent on a sequential computation.
    using namespace std::chrono;

    auto start = steady_clock::now();
    runSequential();
    auto end = steady_clock::now();
    auto elapsed = duration_cast<nanoseconds>(end - start).count();

    if (sequentialRunsMinTime == 0 || elapsed < sequentialRunsMinTime) {
      sequentialRunsMinTime = elapsed;
    }
  }

  void Scheduler::runMultithreaded()
  {
    SCHEDULER_PROFILER_INCREMENT_MULTITHREADED_RUNS_COUNTER;

    ThreadPool& threadPool = getSchedulersThreadPool();
    unsigned int numOfThreads = threadPool.getNumOfThreads();
    std::atomic_size_t chunksGroupIndex = 0;

    for (unsigned int thread = 0; thread < numOfThreads; ++thread) {
      threadPool.async([this, thread, &chunksGroupIndex]() {
        size_t assignedChunksGroup;

        while ((assignedChunksGroup = chunksGroupIndex++) <
               threadEquationsChunks.size()) {
          SCHEDULER_PROFILER_CHUNKS_GROUP_START(thread);
          const auto& chunksGroup = threadEquationsChunks[assignedChunksGroup];

          for (const ThreadEquationsChunk& chunk : chunksGroup) {
            const Equation& equation = chunk.first;
            const auto& ranges = chunk.second;
            equation.function(ranges.data());
          }

          SCHEDULER_PROFILER_CHUNKS_GROUP_STOP(thread);
        }
      });
    }

    threadPool.wait();
  }

  void Scheduler::runMultithreadedWithCalibration()
  {
    // Measure the time spent on a multithreaded computation.
    using namespace std::chrono;

    auto start = steady_clock::now();
    runMultithreaded();
    auto end = steady_clock::now();
    auto elapsed = duration_cast<nanoseconds>(end - start).count();

    if (multithreadedRunsMinTime == 0 || elapsed < multithreadedRunsMinTime) {
      multithreadedRunsMinTime = elapsed;
    }
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
