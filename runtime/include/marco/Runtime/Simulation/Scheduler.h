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

      void createPartitionsGroupsCounters(size_t amount);

      void createPartitionsGroupsTimers(size_t amount);

      void reset() override;

      void print() const override;

    public:
      profiling::Timer addEquation;
      profiling::Timer initialization;
      profiling::Timer run;
      int64_t sequentialRuns{0};
      int64_t multithreadedRuns{0};
      std::vector<int64_t> partitionsGroupsCounters;
      std::vector<std::unique_ptr<profiling::Timer>> partitionsGroups;

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

      // An equation partition is composed of:
      //   - the equation descriptor.
      //   - the ranges information to be passed to the equation function.
      using EquationPartition = std::pair<Equation, std::vector<int64_t>>;

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
          const EquationPartition& equationPartition) const;

      void runSequential();

      void runSequentialWithCalibration();

      void runMultithreaded();

      void runMultithreadedWithCalibration();

    private:
      int64_t identifier{0};
      bool initialized{false};
      std::vector<Equation> equations;

      // The list of equation partitions to be executed in case of sequential
      // execution policy.
      // The information is computed only once during the initialization.
      std::vector<EquationPartition> sequentialSchedule;

      // A group of equation partitions.
      using EquationsGroup = std::vector<EquationPartition>;

      // The list of equations groups the threads will process. Each thread
      // processes one group at a time.
      // The information is computed only once during the initialization.
      std::vector<EquationsGroup> multithreadedSchedule;

      int64_t runsCounter{0};
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
