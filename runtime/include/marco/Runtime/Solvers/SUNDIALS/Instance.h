#ifndef MARCO_RUNTIME_SOLVERS_SUNDIALS_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_SUNDIALS_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Support/Mangling.h"
#include "marco/Runtime/Modeling/MultidimensionalRange.h"
#include "marco/Runtime/Multithreading/ThreadPool.h"
#include <vector>

namespace marco::runtime::sundials
{
  class VariableIndicesIterator;

  /// The list of dimensions of an array variable.
  class VariableDimensions
  {
    private:
      using Container = std::vector<uint64_t>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      explicit VariableDimensions(size_t rank);

      [[nodiscard]] size_t rank() const;

      uint64_t& operator[](size_t index);
      const uint64_t& operator[](size_t index) const;

      /// @name Dimensions iterators
      /// {

      [[nodiscard]] const_iterator begin() const;
      [[nodiscard]] const_iterator end() const;

      /// }
      /// @name Indices iterators
      /// {

      [[nodiscard]] VariableIndicesIterator indicesBegin() const;
      [[nodiscard]] VariableIndicesIterator indicesEnd() const;

      /// }

    private:
      /// Check that all the dimensions have been correctly initialized.
      [[nodiscard, maybe_unused]] bool isValid() const;

    private:
      Container dimensions;
  };

  /// This class is used to iterate on all the possible combination of indices
  /// of a variable.
  class VariableIndicesIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = const uint64_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = const uint64_t**;
      using reference = const uint64_t*&;

      ~VariableIndicesIterator();

      static VariableIndicesIterator begin(
          const VariableDimensions& dimensions);

      static VariableIndicesIterator end(
          const VariableDimensions& dimensions);

      bool operator==(const VariableIndicesIterator& it) const;

      bool operator!=(const VariableIndicesIterator& it) const;

      VariableIndicesIterator& operator++();
      VariableIndicesIterator operator++(int);

      const uint64_t* operator*() const;

    private:
      explicit VariableIndicesIterator(const VariableDimensions& dimensions);

      void fetchNext();

    private:
      uint64_t* indices;
      const VariableDimensions* dimensions;
  };

  using Equation = uint64_t;
  using Variable = uint64_t;

  /// Signature of variable getter functions.
  /// The 1st argument is a pointer to the indices list.
  /// The result is the scalar value.
  using VariableGetter = double(*)(const uint64_t*);

  /// Signature of variable setter functions.
  /// The 1st argument is the value to be set.
  /// The 2nd argument is a pointer to the indices list.
  using VariableSetter = void(*)(double, const uint64_t*);

  /// Signature of the access functions.
  /// The 1st argument is the rank of the equation.
  /// The 2nd argument is a pointer to the list of equation indices.
  /// The 3rd argument is a pointer to the list of results.
  using AccessFunction = void(*)(const int64_t*, uint64_t*);

  /// A column of the Jacobian matrix.
  /// The first element represents the array variable with respect to which the
  /// partial derivative has to be computed. The second element represents the
  /// indices of the scalar variable.
  using JacobianColumn = std::pair<Variable, std::vector<uint64_t>>;

  using VarAccessList = std::vector<std::pair<Variable, AccessFunction>>;

  /// Check if SUNDIALS function returned NULL pointer (no memory allocated).
  bool checkAllocation(void* retval, const char* functionName);

  void printIndices(const std::vector<int64_t>& indices);

  void printIndices(const std::vector<uint64_t>& indices);

  bool advanceVariableIndices(
      uint64_t* indices, const VariableDimensions& dimensions);

  bool advanceVariableIndices(
      std::vector<uint64_t>& indices, const VariableDimensions& dimensions);

  bool advanceVariableIndicesUntil(
      std::vector<uint64_t>& indices,
      const VariableDimensions& dimensions,
      const std::vector<uint64_t>& end);

  /// Given an array of indices and the dimensions of an equation, increase the
  /// indices within the induction bounds of the equation. Return false if the
  /// indices exceed the equation bounds, which means the computation has
  /// finished, true otherwise.
  bool advanceEquationIndices(
      int64_t* indices, const MultidimensionalRange& ranges);

  bool advanceEquationIndices(
      std::vector<int64_t>& indices, const MultidimensionalRange& ranges);

  bool advanceEquationIndicesUntil(
      std::vector<int64_t>& indices,
      const MultidimensionalRange& ranges,
      const std::vector<int64_t>& end);

  /// Get the flat index corresponding to a multidimensional access.
  /// Example:
  ///   x[d1][d2][d3]
  ///   x[i][j][k] -> x[i * d2 * d3 + j * d3 + k]
  uint64_t getVariableFlatIndex(
      const VariableDimensions& dimensions,
      const uint64_t* indices);

  uint64_t getVariableFlatIndex(
      const VariableDimensions& dimensions,
      const std::vector<uint64_t>& indices);

  uint64_t getEquationFlatIndex(
      const std::vector<int64_t>& indices,
      const MultidimensionalRange& ranges);

  void getEquationIndicesFromFlatIndex(
      uint64_t flatIndex,
      std::vector<int64_t>& result,
      const MultidimensionalRange& ranges);
}

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_SUNDIALS_INSTANCE_H
