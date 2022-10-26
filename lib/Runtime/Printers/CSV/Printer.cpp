#include "marco/Runtime/Printers/CSV/Printer.h"
#include "marco/Runtime/Printers/CSV/CLI.h"
#include "marco/Runtime/Printers/CSV/Options.h"
#include "marco/Runtime/Printers/CSV/Profiler.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include <cassert>
#include <iomanip>
#include <iostream>

using namespace ::marco::runtime;
using namespace ::marco::runtime::printing;

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

static void printHeader(const Simulation& simulation)
{
  PRINT_PROFILER_STRING_START;
  std::cout << '"' << "time" << '"';
  PRINT_PROFILER_STRING_STOP;

  for (const int64_t& var : simulation.variablesPrintOrder) {
    if (simulation.variablesPrintableIndices[var].empty()) {
      // The variable has no printable indices.
      continue;
    }

    int64_t rank = simulation.variablesRanks[var];
    int64_t derOrder = simulation.derOrders[var];
    int64_t baseVar = var;

    for (int64_t i = 0; i < derOrder; ++i) {
      baseVar = simulation.derivativesMap[baseVar];
    }

    assert(baseVar != -1);
    char* name = simulation.variablesNames[baseVar];

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

      for (const auto& range : simulation.variablesPrintableIndices[var]) {
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

static void printValues(const Simulation& simulation)
{
  auto& options = printOptions();

  if (options.scientificNotation) {
    std::cout << std::scientific;
  } else {
    std::cout << std::fixed << std::setprecision(options.precision);
  }

  double time = getTime(simulation.getData());

  PRINT_PROFILER_FLOAT_START;
  std::cout << time;
  PRINT_PROFILER_FLOAT_STOP;

  for (const int64_t& var : simulation.variablesPrintOrder) {
    if (simulation.variablesPrintableIndices[var].empty()) {
      // The variable has no printable indices.
      continue;
    }

    int64_t rank = simulation.variablesRanks[var];

    if (rank == 0) {
      // Print the scalar variable.
      double value = getVariableValue(simulation.getData(), var, nullptr);

      PRINT_PROFILER_STRING_START;
      std::cout << ',';
      PRINT_PROFILER_STRING_STOP;

      PRINT_PROFILER_FLOAT_START;
      std::cout << value;
      PRINT_PROFILER_FLOAT_STOP;
    } else {
      // Print the components of the array variable.
      for (const auto& range : simulation.variablesPrintableIndices[var]) {
        auto beginIt = MultidimensionalRangeIterator::begin(range);
        auto endIt = MultidimensionalRangeIterator::end(range);

        for (auto it = beginIt; it != endIt; ++it) {
          double value = getVariableValue(simulation.getData(), var, *it);

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

namespace marco::runtime::printing
{
  CSVPrinter::CSVPrinter(Simulation* simulation)
      : Printer(simulation)
  {
  }

  std::unique_ptr<cli::Category> CSVPrinter::getCLIOptions()
  {
    return std::make_unique<CommandLineOptions>();
  }

  void CSVPrinter::simulationBegin()
  {
    SIMULATION_PROFILER_PRINTING_START;
    ::printHeader(*getSimulation());
    SIMULATION_PROFILER_PRINTING_STOP;
  }

  void CSVPrinter::printValues()
  {
    SIMULATION_PROFILER_PRINTING_START;
    ::printValues(*getSimulation());
    SIMULATION_PROFILER_PRINTING_STOP;
  }

  void CSVPrinter::simulationEnd()
  {
    // Do nothing.
  }
}

namespace marco::runtime
{
  std::unique_ptr<Printer> getPrinter(Simulation* simulation)
  {
    return std::make_unique<printing::CSVPrinter>(simulation);
  }
}
