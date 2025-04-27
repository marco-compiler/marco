#ifndef MARCO_MODELING_DIMENSIONACCESSDIMENSION_H
#define MARCO_MODELING_DIMENSIONACCESSDIMENSION_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling {
class DimensionAccessDimension : public DimensionAccess {
public:
  DimensionAccessDimension(mlir::MLIRContext *context, uint64_t dimension);

  static bool classof(const DimensionAccess *obj) {
    return obj->getKind() == DimensionAccess::Kind::Dimension;
  }

  [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

  [[nodiscard]] bool operator==(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator==(const DimensionAccessDimension &other) const;

  [[nodiscard]] bool operator!=(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator!=(const DimensionAccessDimension &other) const;

  llvm::raw_ostream &dump(llvm::raw_ostream &os,
                          const llvm::DenseMap<const IndexSet *, uint64_t>
                              &iterationSpacesIds) const override;

  void collectIterationSpaces(
      llvm::SetVector<const IndexSet *> &iterationSpaces) const override;

  void collectIterationSpaces(
      llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
      llvm::DenseMap<const IndexSet *, llvm::SetVector<uint64_t>>
          &dependentDimensions) const override;

  [[nodiscard]] bool isConstant() const override;

  [[nodiscard]] bool isAffine() const override;

  [[nodiscard]] mlir::AffineExpr getAffineExpr() const override;

  [[nodiscard]] mlir::AffineExpr
  getAffineExpr(unsigned int numOfDimensions,
                FakeDimensionsMap &fakeDimensionsMap) const override;

  [[nodiscard]] IndexSet map(const Point &point,
                             llvm::DenseMap<const IndexSet *, Point>
                                 &currentIndexSetsPoint) const override;

  [[nodiscard]] uint64_t getDimension() const;

private:
  uint64_t dimension;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_DIMENSIONACCESSDIMENSION_H
