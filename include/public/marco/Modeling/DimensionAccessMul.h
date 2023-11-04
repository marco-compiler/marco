#ifndef MARCO_MODELING_DIMENSIONACCESSMUL_H
#define MARCO_MODELING_DIMENSIONACCESSMUL_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessMul : public DimensionAccess
  {
    public:
      DimensionAccessMul(
          mlir::MLIRContext* context,
          std::unique_ptr<DimensionAccess> first,
          std::unique_ptr<DimensionAccess> second);

      DimensionAccessMul(const DimensionAccessMul& other);

      DimensionAccessMul(DimensionAccessMul&& other) noexcept;

      ~DimensionAccessMul() override;

      DimensionAccessMul& operator=(const DimensionAccessMul& other);

      DimensionAccessMul& operator=(DimensionAccessMul&& other) noexcept;

      friend void swap(
          DimensionAccessMul& first, DimensionAccessMul& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Mul;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(const DimensionAccessMul& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(const DimensionAccessMul& other) const;

      llvm::raw_ostream& dump(
          llvm::raw_ostream& os,
          const llvm::DenseMap<const IndexSet*, uint64_t>& iterationSpacesIds)
          const override;

      void collectIterationSpaces(
          llvm::DenseSet<const IndexSet*>& iterationSpaces) const override;

      void collectIterationSpaces(
          llvm::SmallVectorImpl<const IndexSet*>& iterationSpaces,
          llvm::DenseMap<
              const IndexSet*,
              llvm::DenseSet<uint64_t>>& dependentDimensions) const override;

      [[nodiscard]] bool isAffine() const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr() const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] IndexSet map(
          const Point& point,
          llvm::DenseMap<
              const IndexSet*, Point>& currentIndexSetsPoint) const override;

      [[nodiscard]] DimensionAccess& getFirst();

      [[nodiscard]] const DimensionAccess& getFirst() const;

      [[nodiscard]] DimensionAccess& getSecond();

      [[nodiscard]] const DimensionAccess& getSecond() const;

    private:
      std::unique_ptr<DimensionAccess> first;
      std::unique_ptr<DimensionAccess> second;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSMUL_H
