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

      DimensionAccessDimension(DimensionAccessDimension&& other);

      ~DimensionAccessDimension() override;

      DimensionAccessDimension& operator=(const DimensionAccessDimension& other);

      DimensionAccessDimension& operator=(DimensionAccessDimension&& other);

      friend void swap(
          DimensionAccessDimension& first, DimensionAccessDimension& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Dimension;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessDimension& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessDimension& other) const;

      bool isAffine() const override;

      mlir::AffineExpr getAffineExpr() const override;

      mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      IndexSet map(const Point& point) const override;

      uint64_t getDimension() const;

    private:
      uint64_t dimension;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSDIMENSION_H
