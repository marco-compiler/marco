#ifndef MARCO_MODELING_MCIMIMPL_H
#define MARCO_MODELING_MCIMIMPL_H

#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include <map>
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

          bool operator<(const Delta& other) const;

          long operator[](size_t index) const;

          size_t size() const;

          Delta inverse() const;

        private:
          std::vector<Point::data_type> offsets;
      };

      class MCIMElement
      {
        public:
          MCIMElement();

          MCIMElement(IndexSet keys);

          const IndexSet& getKeys() const;

          void addKeys(IndexSet newKeys);

          IndexSet getValues(const Delta& delta) const;

          MCIMElement inverse(const Delta& delta) const;

        private:
          IndexSet keys;
      };

      Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);
      
      Impl(IndexSet equationRanges, IndexSet variableRanges);

      virtual ~Impl();

      bool operator==(const MCIM::Impl& rhs) const;

      bool operator!=(const MCIM::Impl& rhs) const;

      std::unique_ptr<MCIM::Impl> clone();

      const IndexSet& getEquationRanges() const;

      const IndexSet& getVariableRanges() const;

      IndicesIterator indicesBegin() const;

      IndicesIterator indicesEnd() const;

      virtual MCIM::Impl& operator+=(const MCIM::Impl& rhs);

      virtual MCIM::Impl& operator-=(const MCIM::Impl& rhs);

      void apply(const AccessFunction& access);

      void apply(const MultidimensionalRange& equations, const AccessFunction& access);

      void apply(const IndexSet& equations, const AccessFunction& access);

      bool get(const Point& equation, const Point& variable) const;

      void set(const Point& equation, const Point& variable);

      void unset(const Point& equation, const Point& variable);

      bool empty() const;

      void clear();

      IndexSet flattenRows() const;

      IndexSet flattenColumns() const;

      std::unique_ptr<MCIM::Impl> filterRows(const IndexSet& filter) const;

      std::unique_ptr<MCIM::Impl> filterColumns(const IndexSet& filter) const;

      std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const;

    private:
      bool apply(
          const IndexSet& equations,
          const AccessFunctionRotoTranslation& accessFunction);

      bool apply(
          const MultidimensionalRange& equations,
          const AccessFunctionRotoTranslation& accessFunction);

      Delta getDelta(const Point& equation, const Point& variable) const;

      Delta getDelta(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const;

      const Point& getKey(const Point& equation, const Point& variable) const;

      const MultidimensionalRange& getKey(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const;

      void set(const MultidimensionalRange& equations, const MultidimensionalRange& variables);

      void add(IndexSet equations, Delta delta);

    private:
      IndexSet equationRanges;
      IndexSet variableRanges;

      std::map<Delta, MCIMElement> groups;
  };
}

#endif // MARCO_MODELING_MCIMIMPL_H
