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
      static DimensionAccess constant(Point::data_type position);

      static DimensionAccess relative(
          unsigned int inductionVariableIndex,
          Point::data_type relativePosition);

      // TODO test
      bool operator==(const DimensionAccess& other) const;

      // TODO test
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

      // TODO test
      bool operator==(const AccessFunction& other) const;

      // TODO test
      bool operator!=(const AccessFunction& other) const;

      // TODO test
      static AccessFunction identity(size_t dimensionality);

      const DimensionAccess& operator[](size_t index) const;

      AccessFunction combine(const AccessFunction& other) const;

      DimensionAccess combine(const DimensionAccess& other) const;

      /// Get the number of single dimension accesses.
      size_t size() const;

      const_iterator begin() const;

      const_iterator end() const;

      bool isIdentity() const;

      bool isInvertible() const;

      AccessFunction inverse() const;

      /// Apply the access function to the equation indexes, in order
      /// to obtain the accessed variable.
      ///
      /// @param equationIndexes  equation indexes
      /// @return accessed scalar variable
      Point map(const Point& equationIndexes) const;

      /// Apply the access function to a range, in order to obtain
      /// the mapped range.
      MultidimensionalRange map(const MultidimensionalRange& range) const;

      IndexSet map(const IndexSet& indexes) const;

      MultidimensionalRange inverseMap(const MultidimensionalRange& range) const;

      IndexSet inverseMap(const IndexSet& indexes) const;

      /// Apply the inverse of the access function to a set of indices.
      /// If the access function is not invertible, then the inverse indices are determined
      /// starting from a parent set.
      ///
      /// @param indices         indexes to be inverted
      /// @param parentIndices   parent index set
      /// @return indexes mapping to accessIndexes when accessFunction is applied to them
      IndexSet inverseMap(const IndexSet& indices, const IndexSet& parentIndexes) const;

    private:
      Container functions;
  };

  std::ostream& operator<<(std::ostream& stream, const AccessFunction& obj);
}

#endif  // MARCO_MODELING_ACCESSFUNCTION_H
