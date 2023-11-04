#ifndef MARCO_MODELING_DIMENSIONACCESSADD_H
#define MARCO_MODELING_DIMENSIONACCESSADD_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessAdd : public DimensionAccess
  {
    public:
      DimensionAccessAdd(
          mlir::MLIRContext* context,
          std::unique_ptr<DimensionAccess> first,
          std::unique_ptr<DimensionAccess> second);

      DimensionAccessAdd(const DimensionAccessAdd& other);

      DimensionAccessAdd(DimensionAccessAdd&& other) noexcept;

      ~DimensionAccessAdd() override;

      DimensionAccessAdd& operator=(const DimensionAccessAdd& other);

      DimensionAccessAdd& operator=(DimensionAccessAdd&& other) noexcept;

      friend void swap(DimensionAccessAdd& first, DimensionAccessAdd& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Add;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(const DimensionAccessAdd& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(const DimensionAccessAdd& other) const;

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

#endif // MARCO_MODELING_DIMENSIONACCESSADD_H
