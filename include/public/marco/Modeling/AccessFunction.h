#ifndef MARCO_MODELING_ACCESSFUNCTION_H
#define MARCO_MODELING_ACCESSFUNCTION_H

#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

namespace marco::modeling
{
  /// Single dimension access.
  ///
  /// The access can either be constant or with an offset with respect
  /// to an induction variable.
  class DimensionAccess
  {
    private:
      DimensionAccess(
          bool constantAccess,
          Point::data_type position,
          unsigned int inductionVariableIndex = 0);

    public:
      /// Get an access to a fixed position.
      static DimensionAccess constant(Point::data_type position);

      /// Get an access that is relative to an induction variable by an offset.
      static DimensionAccess relative(
          unsigned int inductionVariableIndex,
          Point::data_type relativePosition);

      bool operator==(const DimensionAccess& other) const;

      bool operator!=(const DimensionAccess& other) const;

      Point::data_type operator()(const Point& equationIndexes) const;

      Range operator()(const MultidimensionalRange& range) const;

      bool isConstantAccess() const;

      Point::data_type getPosition() const;

      Point::data_type getOffset() const;

      unsigned int getInductionVariableIndex() const;

      Point::data_type map(const Point& equationIndexes) const;

      /// Get the mapped dimension access.
      /// The input must be a multidimensional range, as the single dimension
      /// access may refer to any of the range dimensions.
      ///
      /// @param range  multidimensional range
      /// @return mapped mono-dimensional range
      Range map(const MultidimensionalRange& range) const;

    private:
      bool constantAccess;
      Point::data_type position;
      unsigned int inductionVariableIndex;
  };

  std::ostream& operator<<(std::ostream& stream, const DimensionAccess& obj);

  /// The access function describes how an array variable is accessed.
  class AccessFunction
  {
    private:
      using Container = llvm::SmallVector<DimensionAccess, 3>;

    public:
      using const_iterator = Container::const_iterator;

      AccessFunction(llvm::ArrayRef<DimensionAccess> functions);

      bool operator==(const AccessFunction& other) const;

      bool operator!=(const AccessFunction& other) const;

      /// Get the identity access for a given rank.
      static AccessFunction identity(size_t dimensionality);

      /// Get the access used for a given dimension.
      const DimensionAccess& operator[](size_t index) const;

      /// Combine the access function with another one.
      ///
      /// Examples:
      ///   [i1 + 3][i0 - 2] * [i1 + 6][i0 - 1] = [i0 + 4][i1 + 2]
      ///   [i1 + 3][i1 - 2] * [5][i0 - 1] = [5][i1 + 2]
      ///   [5][i0 + 3] * [i0 + 1][i1 - 1] = [6][i0 + 2]
      AccessFunction combine(const AccessFunction& other) const;

      /// Combine the access function to another single dimension access.
      ///
      /// Examples:
      ///   [i1 + 3][i0 - 2] * [i1 + 6] = [i0 + 4]
      ///   [i0 + 3][i1 - 2] * [5] = [5]
      ///   [5][i0 + 3] * [i0 + 1] = [6]
      DimensionAccess combine(const DimensionAccess& other) const;

      /// Get the number of single dimension accesses.
      size_t size() const;

      /// @name Accesses iterators
      /// {

      const_iterator begin() const;

      const_iterator end() const;

      /// }

      /// Check if the access function has an identity layout, that is if the
      /// i-th dimension accesses the i-th induction variable with offset 0
      /// (e.g. [i0][i1][i2]).
      bool isIdentity() const;

      /// Check whether the access function is invertible.
      /// An access function is invertible if all the available induction
      /// variables are used.
      bool isInvertible() const;

      /// Get the inverse access.
      /// The function must be invertible.
      AccessFunction inverse() const;

      /// Apply the access function to the equation indices, in order to obtain
      /// the accessed variable.
      Point map(const Point& equationIndices) const;

      /// Apply the access function to a range, in order to obtain the mapped
      /// range.
      MultidimensionalRange map(const MultidimensionalRange& range) const;

      /// Apply the access function to an index set, in order to obtain the
      /// mapped indices.
      IndexSet map(const IndexSet& indices) const;

      /// Compute the inverse and apply it to a range of indices.
      /// The access function must be invertible.
      MultidimensionalRange inverseMap(const MultidimensionalRange& range) const;

      /// Compute the inverse and apply it to an index set.
      /// The access function must be invertible.
      IndexSet inverseMap(const IndexSet& indices) const;

      /// Apply the inverse of the access function to a set of indices.
      /// If the access function is not invertible, then the inverse indices
      /// are determined starting from a parent set.
      IndexSet inverseMap(const IndexSet& accessedIndices, const IndexSet& parentIndices) const;

    private:
      Container functions;
  };

  std::ostream& operator<<(std::ostream& stream, const AccessFunction& obj);
}

#endif  // MARCO_MODELING_ACCESSFUNCTION_H
