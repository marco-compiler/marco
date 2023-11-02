#ifndef MARCO_MODELING_DIMENSIONACCESSINDICES_H
#define MARCO_MODELING_DIMENSIONACCESSINDICES_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/IndexSet.h"

namespace marco::modeling
{
  class DimensionAccessIndices : public DimensionAccess
  {
    public:
      DimensionAccessIndices(
          mlir::MLIRContext* context,
          std::shared_ptr<IndexSet> space,
          uint64_t dimension);

      DimensionAccessIndices(const DimensionAccessIndices& other);

      DimensionAccessIndices(DimensionAccessIndices&& other) noexcept;

      ~DimensionAccessIndices() override;

      DimensionAccessIndices& operator=(const DimensionAccessIndices& other);

      DimensionAccessIndices& operator=(
          DimensionAccessIndices&& other) noexcept;

      friend void swap(
          DimensionAccessIndices& first, DimensionAccessIndices& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Kind::Indices;
      }

      [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

      [[nodiscard]] bool operator==(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator==(const DimensionAccessIndices& other) const;

      [[nodiscard]] bool operator!=(
          const DimensionAccess& other) const override;

      [[nodiscard]] bool operator!=(const DimensionAccessIndices& other) const;

      llvm::raw_ostream& dump(
          llvm::raw_ostream& os,
          const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds)
          const override;

      void collectIndexSets(
          llvm::SmallVectorImpl<IndexSet*>& indexSets) const override;

      [[nodiscard]] mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] IndexSet map(
          const Point& point,
          const FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] IndexSet& getIndices();

      [[nodiscard]] const IndexSet& getIndices() const;

    private:
      std::shared_ptr<IndexSet> space;
      uint64_t dimension;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSINDICES_H
