#ifndef MARCO_MODELING_DIMENSIONACCESSRANGE_H
#define MARCO_MODELING_DIMENSIONACCESSRANGE_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/Range.h"

namespace marco::modeling
{
  class DimensionAccessRange : public DimensionAccess
  {
    public:
      DimensionAccessRange(
          mlir::MLIRContext* context, Range range);

      DimensionAccessRange(const DimensionAccessRange& other);

      DimensionAccessRange(DimensionAccessRange&& other) noexcept;

      ~DimensionAccessRange() override;

      DimensionAccessRange& operator=(const DimensionAccessRange& other);

      DimensionAccessRange& operator=(DimensionAccessRange&& other) noexcept;

      friend void swap(
          DimensionAccessRange& first, DimensionAccessRange& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Range;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(const DimensionAccessRange& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(const DimensionAccessRange& other) const;

      llvm::raw_ostream& dump(
          llvm::raw_ostream& os,
          const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds)
          const override;

      void collectIndexSets(
          llvm::SmallVectorImpl<IndexSet*>& indexSets) const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] IndexSet map(const Point& point) const override;

      [[nodiscard]] Range& getRange();

      [[nodiscard]] const Range& getRange() const;

    private:
      Range range;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSRANGE_H
