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

      DimensionAccessMul(DimensionAccessMul&& other);

      ~DimensionAccessMul() override;

      DimensionAccessMul& operator=(const DimensionAccessMul& other);

      DimensionAccessMul& operator=(DimensionAccessMul&& other);

      friend void swap(
          DimensionAccessMul& first, DimensionAccessMul& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Mul;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessMul& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessMul& other) const;

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

#endif // MARCO_MODELING_DIMENSIONACCESSMUL_H
