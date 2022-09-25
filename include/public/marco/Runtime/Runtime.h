#ifndef MARCO_RUNTIME_RUNTIME_H
#define MARCO_RUNTIME_RUNTIME_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace marco::runtime
{
  /// Mono-dimensional range in the form [begin, end).
  struct Range
  {
    int64_t begin;
    int64_t end;
  };

  class RangeIterator
  {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = int64_t;
      using difference_type = std::ptrdiff_t;
      using pointer = int64_t*;
      using reference = int64_t&;

      static RangeIterator begin(const Range& range);
      static RangeIterator end(const Range& range);

      bool operator==(const RangeIterator& it) const;

      bool operator!=(const RangeIterator& it) const;

      RangeIterator& operator++();

      RangeIterator operator++(int);

      value_type operator*();

    private:
      RangeIterator(int64_t begin, int64_t end);

    private:
      int64_t current_;
      int64_t end_;
  };

  using MultidimensionalRange = std::vector<Range>;

  class MultidimensionalRangeIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = const int64_t**;
      using reference = const int64_t*&;

      static MultidimensionalRangeIterator begin(const MultidimensionalRange& range);
      static MultidimensionalRangeIterator end(const MultidimensionalRange& range);

      bool operator==(const MultidimensionalRangeIterator& it) const;

      bool operator!=(const MultidimensionalRangeIterator& it) const;

      MultidimensionalRangeIterator& operator++();

      MultidimensionalRangeIterator operator++(int);

      const int64_t* operator*() const;

    private:
      MultidimensionalRangeIterator(const MultidimensionalRange& range, std::function<RangeIterator(const Range&)> initFunction);

      void fetchNext();

    private:
      std::vector<RangeIterator> beginIterators;
      std::vector<RangeIterator> currentIterators;
      std::vector<RangeIterator> endIterators;
      std::vector<int64_t> indices;
  };

  using IndexSet = std::vector<MultidimensionalRange>;

  struct SimulationInfo
  {
    std::vector<char*> variablesNames;
    std::vector<int64_t> variablesRanks;
    std::vector<IndexSet> variablesPrintableIndices;
    std::vector<int64_t> variablesPrintOrder;

    /// Maps each derivative with its derived variable (-1 if the variable is
    /// not a derivative).
    std::vector<int64_t> derivativesMap;

    std::vector<int64_t> derOrders;
  };

  void printHeader(const SimulationInfo& simulationInfo);
  void printValues(void* data, const SimulationInfo& simulationInfo);
}

//===---------------------------------------------------------------------===//
// Functions exported by the runtime library
//===---------------------------------------------------------------------===//

extern "C"
{
  /// Entry point of the simulation. Avoiding a 'main' function inside the
  /// runtime library allows the users to possibly define their own entry
  /// point and perform additional initializations.
  [[maybe_unused]] int runSimulation(int argc, char* argv[]);
}

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  /// Get the current time of the simulation.
  double getCurrentTime(void* data);

  /// Get the name of the compiled Modelica model.
  char* getModelName();

  /// Get the number of variables of the model that is being simulated.
  int64_t getNumOfVariables();

  /// Get the name of a variable.
  char* getVariableName(int64_t var);

  /// Get the rank of a variable.
  int64_t getVariableRank(int64_t var);

  /// Get the number of ranges of indices of a variable that are printable.
  int64_t getVariableNumOfPrintableRanges(int64_t var);

  /// Given a variable, the index of one of its printable multi-dimensional
  /// ranges and its dimension of interest, get the begin index of that
  /// mono-dimensional range.
  int64_t getVariablePrintableRangeBegin(int64_t var, int64_t rangeIndex, int64_t dimension);

  /// Given a variable, the index of one of its printable multi-dimensional
  /// ranges and its dimension of interest, get the end index of that
  /// mono-dimensional range.
  int64_t getVariablePrintableRangeEnd(int64_t var, int64_t rangeIndex, int64_t dimension);

  /// Get the value of a variable.
  double getVariableValue(void* data, int64_t var, const int64_t* indices);

  /// Given the index of a variable, get the index of its derivative, if any.
  /// If the variable has no derivative, the value -1 is returned.
  int64_t getDerivative(int64_t var);
}

#endif // MARCO_RUNTIME_RUNTIME_H
