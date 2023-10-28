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

      DimensionAccessAdd(DimensionAccessAdd&& other);

      ~DimensionAccessAdd() override;

      DimensionAccessAdd& operator=(const DimensionAccessAdd& other);

      DimensionAccessAdd& operator=(DimensionAccessAdd&& other);

      friend void swap(DimensionAccessAdd& first, DimensionAccessAdd& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Add;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessAdd& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessAdd& other) const;

      llvm::raw_ostream& dump(llvm::raw_ostream& os) const override;

      bool isAffine() const override;

      mlir::AffineExpr getAffineExpr() const override;

      mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      IndexSet map(
          const Point& point,
          const FakeDimensionsMap& fakeDimensionsMap) const override;

      DimensionAccess& getFirst();

      const DimensionAccess& getFirst() const;

      DimensionAccess& getSecond();

      const DimensionAccess& getSecond() const;

    private:
      std::unique_ptr<DimensionAccess> first;
      std::unique_ptr<DimensionAccess> second;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSADD_H
