#ifndef MARCO_MODELING_MCIMFLAT_H
#define MARCO_MODELING_MCIMFLAT_H

#include "llvm/ADT/SmallVector.h"
#include "marco/modeling/MCIMImpl.h"

namespace marco::modeling::internal
{
  class FlatMCIM : public MCIM::Impl
  {
    public:
      class Delta
      {
        public:
          using data_type = std::make_unsigned_t<Point::data_type>;

#ifndef WINDOWS_NOSTDLIB
          Delta(data_type key, data_type value);
#else
          Delta(size_t key, size_t value);
#endif

          bool operator==(const Delta& other) const;

          std::make_signed_t<data_type> getValue() const;

          Delta inverse() const;

        private:
          std::make_signed_t<data_type> value;
      };

      class MCIMElement
      {
        public:
          MCIMElement(IndexSet keys, Delta delta);

          const IndexSet& getKeys() const;

          void addKeys(IndexSet newKeys);

          const Delta& getDelta() const;

          IndexSet getValues() const;

          MCIMElement inverse() const;

        private:
          IndexSet keys;
          Delta delta;
      };

      FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

      static bool classof(const MCIM::Impl* obj)
      {
        return obj->getKind() == Flat;
      }

      bool operator==(const MCIM::Impl& rhs) const override;

      bool operator!=(const MCIM::Impl& rhs) const override;

      std::unique_ptr<MCIM::Impl> clone() override;

      MCIM::Impl& operator+=(const MCIM::Impl& rhs) override;

      MCIM::Impl& operator-=(const MCIM::Impl& rhs) override;

      void apply(const AccessFunction& access) override;

      bool get(const Point& equation, const Point& variable) const override;

      void set(const Point& equation, const Point& variable) override;

      void unset(const Point& equation, const Point& variable) override;

      bool empty() const override;

      void clear() override;

      IndexSet flattenRows() const override;

      IndexSet flattenColumns() const override;

      std::unique_ptr<MCIM::Impl> filterRows(const IndexSet& filter) const override;

      std::unique_ptr<MCIM::Impl> filterColumns(const IndexSet& filter) const override;

      std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const override;

    private:
      Point getFlatEquation(const Point& equation) const;

      Point getFlatVariable(const Point& variable) const;

      void add(IndexSet keys, Delta delta);

      llvm::SmallVector<MCIMElement, 3> groups;

      // Stored for faster lookup
      llvm::SmallVector<size_t, 3> equationDimensions;
      llvm::SmallVector<size_t, 3> variableDimensions;
  };
}

#endif // MARCO_MODELING_MCIMFLAT_H
