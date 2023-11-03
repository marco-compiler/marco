#ifndef MARCO_MODELING_DIMENSIONACCESSDIMENSION_H
#define MARCO_MODELING_DIMENSIONACCESSDIMENSION_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessDimension : public DimensionAccess
  {
    public:
      DimensionAccessDimension(mlir::MLIRContext* context, uint64_t dimension);

      DimensionAccessDimension(const DimensionAccessDimension& other);

      DimensionAccessDimension(DimensionAccessDimension&& other) noexcept;

      ~DimensionAccessDimension() override;

      DimensionAccessDimension& operator=(const DimensionAccessDimension& other);

      DimensionAccessDimension& operator=(
          DimensionAccessDimension&& other) noexcept;

      friend void swap(
          DimensionAccessDimension& first, DimensionAccessDimension& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Dimension;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(
          const DimensionAccessDimension& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(
          const DimensionAccessDimension& other) const;

      llvm::raw_ostream& dump(
          llvm::raw_ostream& os,
          const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds)
          const override;

      void collectIndexSets(
          llvm::SmallVectorImpl<IndexSet*>& indexSets) const override;

      [[nodiscard]] bool isAffine() const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr() const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] IndexSet map(const Point& point) const override;

      [[nodiscard]] uint64_t getDimension() const;

    private:
      uint64_t dimension;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSDIMENSION_H
