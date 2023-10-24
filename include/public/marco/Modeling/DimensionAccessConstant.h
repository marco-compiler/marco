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

      DimensionAccessConstant(DimensionAccessConstant&& other);

      ~DimensionAccessConstant() override;

      DimensionAccessConstant& operator=(const DimensionAccessConstant& other);

      DimensionAccessConstant& operator=(DimensionAccessConstant&& other);

      friend void swap(
          DimensionAccessConstant& first, DimensionAccessConstant& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Constant;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessConstant& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessConstant& other) const;

      bool isAffine() const override;

      mlir::AffineExpr getAffineExpr() const override;

      mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      IndexSet map(const Point& point) const override;

      int64_t getValue() const;

    private:
      int64_t value;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSCONSTANT_H
