#ifndef MARCO_RUNTIME_SIMULATION_RUNTIME_H
#define MARCO_RUNTIME_SIMULATION_RUNTIME_H

#include "marco/Runtime/Modeling/IndexSet.h"
#include "marco/Runtime/Printers/Printer.h"
#include "marco/Runtime/Multithreading/ThreadPool.h"
#include <cstdint>
#include <vector>

namespace marco::runtime
{
  class Simulation
  {
    public:
      /// Get the data printer.
      Printer* getPrinter();

      /// Get the data printer.
      const Printer* getPrinter() const;

      /// Set the data printer.
      void setPrinter(Printer* printer);

    public:
      std::vector<char*> variablesNames;
      std::vector<int64_t> variablesRanks;
      std::vector<bool> printableVariables;
      std::vector<IndexSet> variablesPrintableIndices;
      std::vector<int64_t> variablesPrintOrder;

      /// Maps each derivative with its derived variable (-1 if the variable is
      /// not a derivative).
      std::vector<int64_t> derivativesMap;

      std::vector<int64_t> derOrders;

    private:
      Printer* printer;
  };
}

//===---------------------------------------------------------------------===//
// Functions exported by the runtime library
//===---------------------------------------------------------------------===//

extern "C"
{
  /// Entry point of the simulation.
  [[maybe_unused]] int runSimulation(int argc, char* argv[]);
}

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  /// Get the current time of the simulation.
  double getTime();

  /// Set the current time of the simulation.
  void setTime(double startTime);

  /// Get the name of the compiled Modelica model.
  char* getModelName();

  /// Get the number of variables of the model that is being simulated.
  int64_t getNumOfVariables();

  /// Get the name of a variable.
  char* getVariableName(int64_t var);

  /// Get the rank of a variable.
  int64_t getVariableRank(int64_t var);

  /// Get whether the variable is allowed to be printed.
  bool isPrintable(int64_t var);

  /// Get the number of ranges of indices of a variable that are printable.
  int64_t getVariableNumOfPrintableRanges(int64_t var);

  /// Given a variable, the index of one of its printable multi-dimensional
  /// ranges and its dimension of interest, get the begin index of that
  /// mono-dimensional range.
  int64_t getVariablePrintableRangeBegin(
      int64_t var, int64_t rangeIndex, int64_t dimension);

  /// Given a variable, the index of one of its printable multi-dimensional
  /// ranges and its dimension of interest, get the end index of that
  /// mono-dimensional range.
  int64_t getVariablePrintableRangeEnd(
      int64_t var, int64_t rangeIndex, int64_t dimension);

  /// Get the value of a variable.
  double getVariableValue(int64_t var, const int64_t* indices);

  /// Given the index of a variable, get the index of its derivative, if any.
  /// If the variable has no derivative, the value -1 is returned.
  int64_t getDerivative(int64_t var);

  /// Callback function for the beginning 'initial conditions' model solution.
  void icModelBegin();

  /// Callback function for the beginning of the 'initial conditions' model
  /// processing.
  void icModelEnd();

  /// Callback function for the end of the 'initial conditions' model
  /// processing.
  void solveICModel();

  /// Callback function for the beginning of the 'main' model processing.
  void dynamicModelBegin();

  /// Callback function for the end of the 'main' model processing.
  void dynamicModelEnd();
}

#endif // MARCO_RUNTIME_SIMULATION_RUNTIME_H
