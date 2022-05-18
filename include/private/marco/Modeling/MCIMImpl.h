#ifndef MARCO_MODELING_MCIMIMPL_H
#define MARCO_MODELING_MCIMIMPL_H

#include "llvm/Support/Casting.h"
#include "marco/Modeling/MCIM.h"
#include <memory>

namespace marco::modeling::internal
{
  class MCIM::Impl
  {
    public:
      class Delta
      {
        public:
          Delta(const Point& keys, const Point& values);

          Delta(const MultidimensionalRange& keys, const MultidimensionalRange& values);

          bool operator==(const Delta& other) const;

          //size_t getRankDifference() const;

          long operator[](size_t index) const;

          size_t size() const;

          Delta inverse() const;

        private:
          //size_t rankDifference;
          std::vector<Point::data_type> offsets;
      };

      class MCIMElement
      {
        public:
          MCIMElement(IndexSet keys, Delta delta);

          //bool contains(const Point& equation, const Point& variable) const;

          const IndexSet& getKeys() const;

          void addKeys(IndexSet newKeys);

          const Delta& getDelta() const;

          IndexSet getValues() const;

          MCIMElement inverse() const;

        private:
          IndexSet keys;
          Delta delta;
      };

      Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

      bool operator==(const MCIM::Impl& rhs) const;

      bool operator!=(const MCIM::Impl& rhs) const;

      std::unique_ptr<MCIM::Impl> clone();

      const MultidimensionalRange& getEquationRanges() const;

      const MultidimensionalRange& getVariableRanges() const;

      llvm::iterator_range<IndexesIterator> getIndexes() const;

      virtual MCIM::Impl& operator+=(const MCIM::Impl& rhs);

      virtual MCIM::Impl& operator-=(const MCIM::Impl& rhs);

      void apply(const AccessFunction& access);

      bool get(const Point& equation, const Point& variable) const;

      void set(const Point& equation, const Point& variable);

      void set(const MultidimensionalRange& equations, const MultidimensionalRange& variables);

      void unset(const Point& equation, const Point& variable);

      bool empty() const;

      void clear();

      IndexSet flattenRows() const;

      IndexSet flattenColumns() const;

      std::unique_ptr<MCIM::Impl> filterRows(const IndexSet& filter) const;

      std::unique_ptr<MCIM::Impl> filterColumns(const IndexSet& filter) const;

      std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const;

    private:
      Delta getDelta(const Point& equation, const Point& variable) const;

      Delta getDelta(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const;

      const Point& getKey(const Point& equation, const Point& variable) const;

      const MultidimensionalRange& getKey(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const;

      void add(IndexSet equations, Delta delta);

    private:
      MultidimensionalRange equationRanges;
      MultidimensionalRange variableRanges;

      std::vector<MCIMElement> groups;
  };
}

#endif // MARCO_MODELING_MCIMIMPL_H
