#ifndef MARCO_MODELING_DIMENSIONACCESSSUB_H
#define MARCO_MODELING_DIMENSIONACCESSSUB_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessSub : public DimensionAccess
  {
    public:
      DimensionAccessSub(
          mlir::MLIRContext* context,
          std::unique_ptr<DimensionAccess> first,
          std::unique_ptr<DimensionAccess> second);

      DimensionAccessSub(const DimensionAccessSub& other);

      DimensionAccessSub(DimensionAccessSub&& other) noexcept;

      ~DimensionAccessSub() override;

      DimensionAccessSub& operator=(const DimensionAccessSub& other);

      DimensionAccessSub& operator=(DimensionAccessSub&& other) noexcept;

      friend void swap(
          DimensionAccessSub& first, DimensionAccessSub& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Sub;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(const DimensionAccessSub& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(const DimensionAccessSub& other) const;

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

#endif // MARCO_MODELING_DIMENSIONACCESSSUB_H
