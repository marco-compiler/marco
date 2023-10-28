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

      DimensionAccessSub(DimensionAccessSub&& other);

      ~DimensionAccessSub() override;

      DimensionAccessSub& operator=(const DimensionAccessSub& other);

      DimensionAccessSub& operator=(DimensionAccessSub&& other);

      friend void swap(
          DimensionAccessSub& first, DimensionAccessSub& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Sub;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessSub& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessSub& other) const;

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

#endif // MARCO_MODELING_DIMENSIONACCESSSUB_H
