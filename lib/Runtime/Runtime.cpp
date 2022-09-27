#include "marco/Runtime/Runtime.h"
#include "marco/Runtime/CLI.h"
#include "marco/Runtime/IDA.h"
#include "marco/Runtime/Print.h"

using namespace ::marco::runtime;

namespace marco::runtime
{
  RangeIterator::RangeIterator(int64_t begin, int64_t end) : current_(begin), end_(end)
  {
    assert(begin <= end);
  }

  RangeIterator RangeIterator::begin(const Range& range)
  {
    return RangeIterator(range.begin, range.end);
  }

  RangeIterator RangeIterator::end(const Range& range)
  {
    return RangeIterator(range.end, range.end);
  }

  bool RangeIterator::operator==(const RangeIterator& it) const
  {
    return current_ == it.current_ && end_ == it.end_;
  }

  bool RangeIterator::operator!=(const RangeIterator& it) const
  {
    return current_ != it.current_ || end_ != it.end_;
  }

  RangeIterator& RangeIterator::operator++()
  {
    current_ = std::min(current_ + 1, end_);
    return *this;
  }

  RangeIterator RangeIterator::operator++(int)
  {
    auto temp = *this;
    current_ = std::min(current_ + 1, end_);
    return temp;
  }

  int64_t RangeIterator::operator*()
  {
    return current_;
  }

  MultidimensionalRangeIterator::MultidimensionalRangeIterator(const MultidimensionalRange& ranges, std::function<RangeIterator(const Range&)> initFunction)
  {
    for (const auto& range: ranges) {
      beginIterators.push_back(RangeIterator::begin(range));
      auto it = initFunction(range);
      currentIterators.push_back(it);
      endIterators.push_back(RangeIterator::end(range));
      indices.push_back(*it);
    }

    assert(ranges.size() == beginIterators.size());
    assert(ranges.size() == currentIterators.size());
    assert(ranges.size() == endIterators.size());
    assert(ranges.size() == indices.size());
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::begin(const MultidimensionalRange& ranges)
  {
    return MultidimensionalRangeIterator(ranges, [](const Range& range) {
      return RangeIterator::begin(range);
    });
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::end(const MultidimensionalRange& ranges)
  {
    return MultidimensionalRangeIterator(ranges, [](const Range& range) {
      return RangeIterator::end(range);
    });
  }

  bool MultidimensionalRangeIterator::operator==(const MultidimensionalRangeIterator& it) const
  {
    return currentIterators == it.currentIterators;
  }

  bool MultidimensionalRangeIterator::operator!=(const MultidimensionalRangeIterator& it) const
  {
    return currentIterators != it.currentIterators;
  }

  MultidimensionalRangeIterator& MultidimensionalRangeIterator::operator++()
  {
    fetchNext();
    return *this;
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  const int64_t* MultidimensionalRangeIterator::operator*() const
  {
    return indices.data();
  }

  void MultidimensionalRangeIterator::fetchNext()
  {
    size_t size = indices.size();

    auto findIndex = [&]() -> std::pair<bool, size_t> {
      for (size_t i = 0, e = size; i < e; ++i) {
        size_t pos = e - i - 1;

        if (++currentIterators[pos] != endIterators[pos]) {
          return std::make_pair(true, pos);
        }
      }

      return std::make_pair(false, 0);
    };

    std::pair<bool, size_t> index = findIndex();

    if (index.first) {
      size_t pos = index.second;

      indices[pos] = *currentIterators[pos];

      for (size_t i = pos + 1; i < size; ++i) {
        currentIterators[i] = beginIterators[i];
        indices[i] = *currentIterators[i];
      }
    }
  }
}

extern "C"
{
  void* init();

  void* initICSolvers(void* data);
  void* deinitICSolvers(void* data);

  void initMainSolvers(void* data);
  void deinitMainSolvers(void* data);

  void calcIC(void* data);
  void updateNonStateVariables(void* data);
  void updateStateVariables(void* data);
  bool incrementTime(void* data);
  void deinit(void* data);
}

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling.h"
#include <iostream>

namespace
{
  class SimulationProfiler : public Profiler
  {
    public:
      SimulationProfiler() : Profiler("Simulation")
      {
        registerProfiler(*this);
      }

      void reset() override
      {
        commandLineArgs.reset();
        initialization.reset();
        initialConditions.reset();
        nonStateVariables.reset();
        stateVariables.reset();
        printing.reset();
      }

      void print() const override
      {
        std::cerr << "Time spent on command-line arguments processing: " << commandLineArgs.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on initialization: " << initialization.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on initial conditions computation: " << initialConditions.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on non-state variables computation: " << nonStateVariables.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on state variables computation: " << stateVariables.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on values printing: " << printing.totalElapsedTime() << " ms\n";
      }

    public:
      Timer commandLineArgs;
      Timer initialization;
      Timer initialConditions;
      Timer nonStateVariables;
      Timer stateVariables;
      Timer printing;
  };

  SimulationProfiler& profiler()
  {
    static SimulationProfiler obj;
    return obj;
  }
}

  #define PROFILER_ARG_START ::profiler().commandLineArgs.start()
  #define PROFILER_ARG_STOP ::profiler().commandLineArgs.stop()

  #define PROFILER_INIT_START ::profiler().initialization.start()
  #define PROFILER_INIT_STOP ::profiler().initialization.stop()

  #define PROFILER_IC_START ::profiler().initialConditions.start()
  #define PROFILER_IC_STOP ::profiler().initialConditions.stop()

  #define PROFILER_NONSTATEVAR_START ::profiler().nonStateVariables.start()
  #define PROFILER_NONSTATEVAR_STOP ::profiler().nonStateVariables.stop()

  #define PROFILER_STATEVAR_START ::profiler().stateVariables.start()
  #define PROFILER_STATEVAR_STOP ::profiler().stateVariables.stop()

  #define PROFILER_PRINTING_START ::profiler().printing.start()
  #define PROFILER_PRINTING_STOP ::profiler().printing.stop()

#else
  #define PROFILER_DO_NOTHING static_assert(true)

  #define PROFILER_ARG_START PROFILER_DO_NOTHING
  #define PROFILER_ARG_STOP PROFILER_DO_NOTHING

  #define PROFILER_INIT_START PROFILER_DO_NOTHING
  #define PROFILER_INIT_STOP PROFILER_DO_NOTHING

  #define PROFILER_IC_START PROFILER_DO_NOTHING
  #define PROFILER_IC_STOP PROFILER_DO_NOTHING

  #define PROFILER_NONSTATEVAR_START PROFILER_DO_NOTHING
  #define PROFILER_NONSTATEVAR_STOP PROFILER_DO_NOTHING

  #define PROFILER_STATEVAR_START PROFILER_DO_NOTHING
  #define PROFILER_STATEVAR_STOP PROFILER_DO_NOTHING

  #define PROFILER_PRINTING_START PROFILER_DO_NOTHING
  #define PROFILER_PRINTING_STOP PROFILER_DO_NOTHING
#endif

namespace
{
  SimulationInfo runtimeInit()
  {
    #ifdef MARCO_PROFILING
    profilingInit();
    #endif

    SimulationInfo result;

    // Number of array variables of the model (both state and algebraic ones).
    int64_t numOfVariables = getNumOfVariables();

    // Pre-fetch the names.
    result.variablesNames.resize(numOfVariables);

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      result.variablesNames[var] = getVariableName(var);
    }

    // Pre-fetch the ranks.
    result.variablesRanks.resize(numOfVariables);

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      result.variablesRanks[var] = getVariableRank(var);
    }

    // Pre-fetch the printable indices.
    result.variablesPrintableIndices.resize(numOfVariables);
    result.variablesPrintOrder.resize(numOfVariables);

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      int64_t numOfPrintableRanges = getVariableNumOfPrintableRanges(var);
      result.variablesPrintableIndices[var].resize(numOfPrintableRanges);

      for (int64_t range = 0; range < numOfPrintableRanges; ++range) {
        int64_t rank = result.variablesRanks[var];
        result.variablesPrintableIndices[var][range].resize(rank);

        for (int64_t dim = 0; dim < rank; ++dim) {
          result.variablesPrintableIndices[var][range][dim].begin = getVariablePrintableRangeBegin(var, range, dim);
          result.variablesPrintableIndices[var][range][dim].end = getVariablePrintableRangeEnd(var, range, dim);
        }
      }

      result.variablesPrintOrder[var] = var;
    }

    // Pre-fetch the derivatives map.
    result.derivativesMap.resize(numOfVariables);

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      result.derivativesMap[var] = -1;
    }

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      int64_t derivative = getDerivative(var);

      if (derivative != -1) {
        result.derivativesMap[derivative] = var;
      }
    }

    // Compute the derivative order of each variable.
    result.derOrders.resize(numOfVariables);

    for (int64_t var = 0, e = numOfVariables; var < e; ++var) {
      int64_t currentVar = result.derivativesMap[var];
      int64_t order = 0;

      while (currentVar != -1) {
        ++order;
        currentVar = result.derivativesMap[currentVar];
      }

      result.derOrders[var] = order;
    }

    // Determine the print ordering for the variables.
    std::sort(result.variablesPrintOrder.begin(), result.variablesPrintOrder.end(),
              [&](const int64_t& x, const int64_t& y) -> bool {
                int64_t xDerOrder = result.derOrders[x];
                int64_t yDerOrder = result.derOrders[y];

                if (xDerOrder < yDerOrder) {
                  return true;
                }

                if (xDerOrder > yDerOrder) {
                  return false;
                }

                int64_t first = x;
                int64_t second = y;

                if (xDerOrder != 0) {
                  for (int64_t i = 0; i < xDerOrder; ++i) {
                    first = result.derivativesMap[first];
                  }

                  for (int64_t i = 0; i < yDerOrder; ++i) {
                    second = result.derivativesMap[second];
                  }
                }

                return std::string_view(result.variablesNames[first]) < std::string_view(result.variablesNames[second]);
              });

    return result;
  }

  void runtimeDeinit(SimulationInfo& simulationInfo)
  {
    #ifdef MARCO_PROFILING
    printProfilingStats();
    #endif
  }
}

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

namespace
{
  void printHelp()
  {
    std::cout << "Modelica simulation.\n";
    std::cout << "Model: " << getModelName() << "\n";
    std::cout << "Generated with MARCO compiler.\n\n";

    std::cout << "OPTIONS:\n";
    std::cout << "  --help      Display the available options.\n\n";

    auto& cli = getCLI();

    for (size_t i = 0; i < cli.size(); ++i) {
      std::cout << cli[i].getTitle() << "\n";
      cli[i].printCommandLineOptions(std::cout);
      std::cout << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Simulation
//===----------------------------------------------------------------------===//

[[maybe_unused]] int runSimulation(int argc, char* argv[])
{
  // Parse the command-line arguments
  PROFILER_ARG_START;
  auto& cli = getCLI();
  cli += formatting::getCLIOptionsCategory();
  cli += ida::getCLIOptionsCategory();

  argh::parser cmdl(argc, argv);

  if (cmdl["help"]) {
    printHelp();
    return 0;
  }

  for (size_t i = 0; i < cli.size(); ++i) {
    cli[i].parseCommandLineOptions(cmdl);
  }

  PROFILER_ARG_STOP;

  // Initialize the runtime library
  SimulationInfo simulationInfo = runtimeInit();

  // Compute the initial values
  PROFILER_INIT_START;
  void* data = init();
  PROFILER_INIT_STOP;

  // TODO check IDA initialization through static checker
  // Same for IDA step

  initICSolvers(data);

  // Compute the initial conditions
  PROFILER_IC_START;
  calcIC(data);
  PROFILER_IC_STOP;

  deinitICSolvers(data);

  // Print the data header and the initial values
  PROFILER_PRINTING_START;
  printHeader(simulationInfo);
  PROFILER_PRINTING_STOP;

  PROFILER_PRINTING_START;
  printValues(data, simulationInfo);
  PROFILER_PRINTING_STOP;

  initMainSolvers(data);

  bool continueSimulation;

  do {
    // Compute the next state variables
    PROFILER_STATEVAR_START;
    updateStateVariables(data);
    PROFILER_STATEVAR_STOP;

    // Move to the next step
    PROFILER_NONSTATEVAR_START;
    continueSimulation = incrementTime(data);
    updateNonStateVariables(data);
    PROFILER_NONSTATEVAR_STOP;

    // Print the values
    PROFILER_PRINTING_START;
    printValues(data, simulationInfo);
    PROFILER_PRINTING_STOP;
  } while (continueSimulation);

  deinitMainSolvers(data);
  deinit(data);

  // De-initialize the runtime library
  runtimeDeinit(simulationInfo);

  return 0;
}
