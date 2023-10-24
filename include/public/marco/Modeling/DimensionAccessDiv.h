#ifndef MARCO_MODELING_DIMENSIONACCESSDIV_H
#define MARCO_MODELING_DIMENSIONACCESSDIV_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling
{
  class DimensionAccessDiv : public DimensionAccess
  {
    public:
      DimensionAccessDiv(
          mlir::MLIRContext* context,
          std::unique_ptr<DimensionAccess> first,
          std::unique_ptr<DimensionAccess> second);

      DimensionAccessDiv(const DimensionAccessDiv& other);

      DimensionAccessDiv(DimensionAccessDiv&& other);

      ~DimensionAccessDiv() override;

      DimensionAccessDiv& operator=(const DimensionAccessDiv& other);

      DimensionAccessDiv& operator=(DimensionAccessDiv&& other);

      friend void swap(
          DimensionAccessDiv& first, DimensionAccessDiv& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Div;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessDiv& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessDiv& other) const;

      bool isAffine() const override;

      mlir::AffineExpr getAffineExpr() const override;

      mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      DimensionAccess& getFirst();

      const DimensionAccess& getFirst() const;

      DimensionAccess& getSecond();

      const DimensionAccess& getSecond() const;

      IndexSet map(const Point& point) const override;

    private:
      std::unique_ptr<DimensionAccess> first;
      std::unique_ptr<DimensionAccess> second;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSDIV_H
