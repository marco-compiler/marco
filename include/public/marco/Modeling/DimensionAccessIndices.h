#ifndef MARCO_MODELING_DIMENSIONACCESSINDICES_H
#define MARCO_MODELING_DIMENSIONACCESSINDICES_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/IndexSet.h"

namespace marco::modeling
{
  class DimensionAccessIndices : public DimensionAccess
  {
    public:
      DimensionAccessIndices(mlir::MLIRContext* context, IndexSet indices);

      DimensionAccessIndices(const DimensionAccessIndices& other);

      DimensionAccessIndices(DimensionAccessIndices&& other);

      ~DimensionAccessIndices() override;

      DimensionAccessIndices& operator=(const DimensionAccessIndices& other);

      DimensionAccessIndices& operator=(DimensionAccessIndices&& other);

      friend void swap(
          DimensionAccessIndices& first, DimensionAccessIndices& second);

      static bool classof(const DimensionAccess* obj)
      {
        return obj->getKind() == DimensionAccess::Indices;
      }

      std::unique_ptr<DimensionAccess> clone() const override;

      bool operator==(const DimensionAccess& other) const override;

      bool operator==(const DimensionAccessIndices& other) const;

      bool operator!=(const DimensionAccess& other) const override;

      bool operator!=(const DimensionAccessIndices& other) const;

      mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const override;

      IndexSet map(const Point& point) const override;

      IndexSet& getIndices();

      const IndexSet& getIndices() const;

    private:
      IndexSet resultIndices;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESSINDICES_H
