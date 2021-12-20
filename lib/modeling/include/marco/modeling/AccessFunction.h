#ifndef MARCO_MODELING_ACCESSFUNCTION_H
#define MARCO_MODELING_ACCESSFUNCTION_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "MultidimensionalRange.h"

namespace marco::modeling
{
  /**
   * Single dimension access.
   *
   * The access can either be constant or with an offset with respect
   * to an induction variable.
   */
  class DimensionAccess
  {
    private:
      using Point = internal::Point;
      using Range = internal::Range;
      using MultidimensionalRange = internal::MultidimensionalRange;

      DimensionAccess(
          bool constantAccess,
          Point::data_type position,
          unsigned int inductionVariableIndex = 0);

    public:
      static DimensionAccess constant(Point::data_type position);

      static DimensionAccess relative(
          unsigned int inductionVariableIndex,
          Point::data_type relativePosition);

      Point::data_type operator()(const Point& equationIndexes) const;

      /**
       * Get the mapped dimension access.
       * The input must be a multidimensional range, as the single dimension
       * access may refer to any of the range dimensions.
       *
       * @param range  multidimensional range
       * @return mapped mono-dimensional range
       */
      Range operator()(const MultidimensionalRange& range) const;

      bool isConstantAccess() const;

      Point::data_type getPosition() const;

      Point::data_type getOffset() const;

      unsigned int getInductionVariableIndex() const;

    private:
      bool constantAccess;
      Point::data_type position;
      unsigned int inductionVariableIndex;
  };

  /**
   * The access function describes how an array variable is accessed.
   */
  class AccessFunction
  {
    private:
      using Point = internal::Point;
      using MultidimensionalRange = internal::MultidimensionalRange;
      using Container = llvm::SmallVector<DimensionAccess, 3>;

    public:
      using const_iterator = Container::const_iterator;

      AccessFunction(llvm::ArrayRef<DimensionAccess> functions);

      const DimensionAccess& operator[](size_t index) const;

      AccessFunction combine(const AccessFunction& other) const;

      DimensionAccess combine(const DimensionAccess& other) const;

      /**
       * Get the number of single dimension accesses.
       *
       * @return number of accesses
       */
      size_t size() const;

      const_iterator begin() const;

      const_iterator end() const;

      bool isIdentity() const;

      /**
       * Apply the access function to the equation indexes, in order
       * to obtain the accessed variable.
       *
       * @param equationIndexes  equation indexes
       * @return accessed scalar variable
       */
      Point map(const Point& equationIndexes) const;

      /**
       * Apply the access function to a range, in order to obtain
       * the mapped range.
       *
       * @param range  multidimensional range
       * @return mapped multidimensional range
       */
      MultidimensionalRange map(const MultidimensionalRange& range) const;

      AccessFunction invert() const;

    private:
      Container functions;
  };
}

#endif  // MARCO_MODELING_ACCESSFUNCTION_H
