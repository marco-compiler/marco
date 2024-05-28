#ifndef MARCO_RUNTIME_SIMULATION_SCHEDULER_H
#define MARCO_RUNTIME_SIMULATION_SCHEDULER_H

#include "marco/Runtime/Modeling/MultidimensionalRange.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include "marco/Runtime/Profiling/Timer.h"
#include "marco/Runtime/Support/Mangling.h"
#include <cstdint>
#include <vector>

namespace marco::runtime
{
#ifdef MARCO_PROFILING
  class SchedulerProfiler : public profiling::Profiler
  {
    public:
      SchedulerProfiler(int64_t schedulerId);

      void createChunksGroupCounters(size_t amount);

      void createChunksGroupTimers(size_t amount);

      void reset() override;

      void print() const override;

    public:
      profiling::Timer addEquation;
      profiling::Timer initialization;
      profiling::Timer run;
      int64_t sequentialRuns{0};
      int64_t multithreadedRuns{0};
      std::vector<int64_t> chunksGroupsCounters;
      std::vector<std::unique_ptr<profiling::Timer>> chunksGroups;

      mutable std::mutex mutex;
  };
#endif

  class Scheduler
  {
    public:
      using EquationFunction = void(*)(const int64_t*);

      struct Equation
      {
        EquationFunction function;
        MultidimensionalRange indices;
        bool independentIndices;

        Equation(
            EquationFunction function,
            MultidimensionalRange indices,
            bool independentIndices);
      };

      // A chunk of equations to be processed by a thread.
      // A chunk is composed of:
      //   - the equation descriptor.
      //   - the ranges information to be passed to the equation function.
      using ThreadEquationsChunk = std::pair<Equation, std::vector<int64_t>>;

      enum class RunStrategy
      {
        Sequential,
        Multithreaded
      };

      Scheduler();

      void addEquation(
          EquationFunction function,
          uint64_t rank,
          int64_t* ranges,
          bool independentIndices);

      void run();

    private:
      void initialize();

      [[maybe_unused, nodiscard]] bool checkEquationScheduledExactlyOnce(
          const Equation& equation) const;

      [[maybe_unused, nodiscard]] bool checkEquationIndicesExistence(
          const ThreadEquationsChunk& chunk) const;

      void runSequential();

      void runSequentialWithCalibration();

      void runMultithreaded();

      void runMultithreadedWithCalibration();

    private:
      int64_t identifier{0};
      bool initialized{false};
      std::vector<Equation> equations;

      std::vector<ThreadEquationsChunk> sequentialSchedule;

      // The list of chunks the threads will process. Each thread elaborates
      // one chunk at a time.
      // The information is computed only once during the initialization to
      // save time during the simulation.
      std::vector<std::vector<ThreadEquationsChunk>> threadEquationsChunks;

      int64_t runsCounter = 0;
      int64_t sequentialRunsMinTime{0};
      int64_t multithreadedRunsMinTime{0};
      RunStrategy runStrategy{RunStrategy::Sequential};

#ifdef MARCO_PROFILING
      // Profiling.
      std::shared_ptr<SchedulerProfiler> profiler;
#endif
  };
}

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(schedulerCreate, PTR(void))

RUNTIME_FUNC_DECL(schedulerDestroy, void, PTR(void))

RUNTIME_FUNC_DECL(schedulerAddEquation, void, PTR(void), PTR(void), uint64_t, PTR(int64_t), bool)

RUNTIME_FUNC_DECL(schedulerRun, void, PTR(void))

#endif // MARCO_RUNTIME_SIMULATION_SCHEDULER_H
