#ifndef MARCO_MODELING_MCIM_H
#define MARCO_MODELING_MCIM_H

#include "llvm/ADT/iterator_range.h"
#include "marco/modeling/AccessFunction.h"
#include "marco/modeling/IndexSet.h"
#include "marco/modeling/MultidimensionalRange.h"

namespace marco::modeling::internal
{
  /// Multidimensional Compressed Index Map (MCIM).
  /// It replaces the multidimensional incidence matrices in order to achieve O(1) scaling.
  class MCIM
  {
    public:
      class Impl;

      class IndexesIterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = std::pair<Point, Point>;
          using difference_type = std::ptrdiff_t;
          using pointer = std::pair<Point, Point>*;
          using reference = std::pair<Point, Point>&;

          using Iterator = IndexSet::const_iterator;

          IndexesIterator(
              const IndexSet& equationRanges,
              const IndexSet& variableRanges,
              std::function<IndexSet::const_iterator(const IndexSet&)> initFunction);

          bool operator==(const IndexesIterator& it) const;

          bool operator!=(const IndexesIterator& it) const;

          IndexesIterator& operator++();

          IndexesIterator operator++(int);

          value_type operator*() const;

        private:
          void advance();

          Iterator eqCurrentIt;
          Iterator eqEndIt;
          Iterator varBeginIt;
          Iterator varCurrentIt;
          Iterator varEndIt;
      };

      MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);
      MCIM(const IndexSet &equationRanges, const IndexSet &variableRanges);

      MCIM(const MCIM& other);

      MCIM(MCIM&& other);

      ~MCIM();

      MCIM& operator=(const MCIM& other);

      friend void swap(MCIM& first, MCIM& second);

      bool operator==(const MCIM& other) const;

      bool operator!=(const MCIM& other) const;

      const IndexSet& getEquationRanges() const;

      const IndexSet& getVariableRanges() const;

      llvm::iterator_range<IndexesIterator> getIndexes() const;

      MCIM& operator+=(const MCIM& rhs);

      MCIM operator+(const MCIM& rhs) const;

      MCIM& operator-=(const MCIM& rhs);

      MCIM operator-(const MCIM& rhs) const;

      void apply(const AccessFunction& access);

      bool get(const Point& equation, const Point& variable) const;

      void set(const Point& equation, const Point& variable);

      void unset(const Point& equation, const Point& variable);

      bool empty() const;

      void clear();

      IndexSet flattenRows() const;

      IndexSet flattenColumns() const;

      MCIM filterRows(const IndexSet& filter) const;

      MCIM filterColumns(const IndexSet& filter) const;

      std::vector<MCIM> splitGroups() const;

    private:
      MCIM(std::unique_ptr<Impl> impl);

      std::unique_ptr<Impl> impl;
  };

  std::ostream& operator<<(std::ostream& stream, const MCIM& mcim);
}

#endif  // MARCO_MODELING_MCIM_H
