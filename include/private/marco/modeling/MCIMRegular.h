#ifndef MARCO_MODELING_MCIMREGULAR_H
#define MARCO_MODELING_MCIMREGULAR_H

#include "llvm/ADT/SmallVector.h"
#include "MCIMImpl.h"

namespace marco::modeling::internal
{
  class RegularMCIM : public MCIM::Impl
  {
    public:
      class Delta
      {
        public:
          Delta(const Point& keys, const Point& values);

          bool operator==(const Delta& other) const;

          long operator[](size_t index) const;

          size_t size() const;

          Delta inverse() const;

        private:
          llvm::SmallVector<Point::data_type, 3> values;
      };

      class MCIMElement
      {
        public:
          MCIMElement(MCIS keys, Delta delta);

          const MCIS& getKeys() const;

          void addKeys(MCIS newKeys);

          const Delta& getDelta() const;

          MCIS getValues() const;

          MCIMElement inverse() const;

        private:
          MCIS keys;
          Delta delta;
      };

      RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

      static bool classof(const MCIM::Impl* obj)
      {
        return obj->getKind() == Regular;
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

      MCIS flattenRows() const override;

      MCIS flattenColumns() const override;

      std::unique_ptr<MCIM::Impl> filterRows(const MCIS& filter) const override;

      std::unique_ptr<MCIM::Impl> filterColumns(const MCIS& filter) const override;

      std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const override;

    private:
      void set(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes);

      void add(MCIS keys, Delta delta);

      llvm::SmallVector<MCIMElement, 3> groups;
  };
}

#endif // MARCO_MODELING_MCIMREGULAR_H
