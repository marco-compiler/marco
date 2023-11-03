#ifndef MARCO_MODELING_DIMENSIONACCESSCONSTANT_H
#define MARCO_MODELING_DIMENSIONACCESSCONSTANT_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessConstant : public DimensionAccess
  {
    public:
      DimensionAccessConstant(mlir::MLIRContext* context, int64_t value);

      DimensionAccessConstant(const DimensionAccessConstant& other);

      DimensionAccessConstant(DimensionAccessConstant&& other) noexcept;

      ~DimensionAccessConstant() override;

      DimensionAccessConstant& operator=(const DimensionAccessConstant& other);

      DimensionAccessConstant& operator=(
          DimensionAccessConstant&& other) noexcept;

      friend void swap(
          DimensionAccessConstant& first, DimensionAccessConstant& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Constant;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(
          const DimensionAccessConstant& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(
          const DimensionAccessConstant& other) const;

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

      [[nodiscard]] int64_t getValue() const;

    private:
      int64_t value;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSCONSTANT_H
