#include "marco/Runtime/Runtime.h"
#include "marco/Runtime/Print.h"
#include <iomanip>

using namespace ::marco::runtime::formatting;

static void printDerWrapOpening(int64_t order)
{
  for (int64_t i = 0; i < order; ++i) {
    PRINT_PROFILER_STRING_START;
    std::cout << "der(";
    PRINT_PROFILER_STRING_STOP;
  }
}

static void printDerWrapClosing(int64_t order)
{
  for (int64_t i = 0; i < order; ++i) {
    PRINT_PROFILER_STRING_START;
    std::cout << ')';
    PRINT_PROFILER_STRING_STOP;
  }
}

static void printName(char* name, int64_t rank, const int64_t* indices)
{
  PRINT_PROFILER_STRING_START;
  std::cout << name;
  PRINT_PROFILER_STRING_STOP;

  if (rank != 0) {
    assert(indices != nullptr);
    std::cout << '[';

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dim != 0) {
        PRINT_PROFILER_STRING_START;
        std::cout << ',';
        PRINT_PROFILER_STRING_STOP;
      }

      // Modelica arrays are 1-based, so increment the printed index by one.
      int64_t index = indices[dim] + 1;

      PRINT_PROFILER_INT_START;
      std::cout << index;
      PRINT_PROFILER_INT_STOP;
    }

    std::cout << ']';
  }
}

namespace marco::runtime
{
  void printHeader(const SimulationInfo& simulationInfo)
  {
    PRINT_PROFILER_STRING_START;
    std::cout << '"' << "time" << '"';
    PRINT_PROFILER_STRING_STOP;

    for (const int64_t& var : simulationInfo.variablesPrintOrder) {
      if (simulationInfo.variablesPrintableIndices[var].empty()) {
        // The variable has no printable indices.
        continue;
      }

      int64_t rank = simulationInfo.variablesRanks[var];
      int64_t derOrder = simulationInfo.derOrders[var];
      int64_t baseVar = var;

      for (int64_t i = 0; i < derOrder; ++i) {
        baseVar = simulationInfo.derivativesMap[baseVar];
      }

      assert(baseVar != -1);
      char* name = simulationInfo.variablesNames[baseVar];

      if (rank == 0) {
        // Print only the variable name.
        PRINT_PROFILER_STRING_START;
        std::cout << ',' << '"';
        PRINT_PROFILER_STRING_STOP;

        printDerWrapOpening(derOrder);
        printName(name, 0, nullptr);
        printDerWrapClosing(derOrder);

        PRINT_PROFILER_STRING_START;
        std::cout << '"';
        PRINT_PROFILER_STRING_STOP;
      } else {
        // Print the name of the array and the indices, for each possible
        // combination of printable indices.

        for (const auto& range : simulationInfo.variablesPrintableIndices[var]) {
          auto beginIt = MultidimensionalRangeIterator::begin(range);
          auto endIt = MultidimensionalRangeIterator::end(range);

          for (auto it = beginIt; it != endIt; ++it) {
            PRINT_PROFILER_STRING_START;
            std::cout << ',' << '"';
            PRINT_PROFILER_STRING_STOP;

            printDerWrapOpening(derOrder);
            printName(name, rank, *it);
            printDerWrapClosing(derOrder);

            PRINT_PROFILER_STRING_START;
            std::cout << '"';
            PRINT_PROFILER_STRING_STOP;
          }
        }
      }
    }

    PRINT_PROFILER_STRING_START;
    std::cout << std::endl;
    PRINT_PROFILER_STRING_STOP;
  }

  void printValues(void* data, const SimulationInfo& simulationInfo)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      std::cout << std::scientific;
    } else {
      std::cout << std::fixed << std::setprecision(config.precision);
    }

    double time = getCurrentTime(data);

    PRINT_PROFILER_FLOAT_START;
    std::cout << time;
    PRINT_PROFILER_FLOAT_STOP;

    for (const int64_t& var : simulationInfo.variablesPrintOrder) {
      if (simulationInfo.variablesPrintableIndices[var].empty()) {
        // The variable has no printable indices.
        continue;
      }

      int64_t rank = simulationInfo.variablesRanks[var];

      if (rank == 0) {
        // Print the scalar variable.
        double value = getVariableValue(data, var, nullptr);
        if (value == 0)
          value = 0;

        PRINT_PROFILER_STRING_START;
        std::cout << ',';
        PRINT_PROFILER_STRING_STOP;

        PRINT_PROFILER_FLOAT_START;
        std::cout << value;
        PRINT_PROFILER_FLOAT_STOP;
      } else {
        // Print the components of the array variable.
        for (const auto& range : simulationInfo.variablesPrintableIndices[var]) {
          auto beginIt = MultidimensionalRangeIterator::begin(range);
          auto endIt = MultidimensionalRangeIterator::end(range);

          for (auto it = beginIt; it != endIt; ++it) {
            double value = getVariableValue(data, var, *it);
            if (value == 0)
              value = 0;

            PRINT_PROFILER_STRING_START;
            std::cout << ',';
            PRINT_PROFILER_STRING_STOP;

            PRINT_PROFILER_FLOAT_START;
            std::cout << value;
            PRINT_PROFILER_FLOAT_STOP;
          }
        }
      }
    }

    PRINT_PROFILER_STRING_START;
    std::cout << std::endl;
    PRINT_PROFILER_STRING_STOP;
  }
}
