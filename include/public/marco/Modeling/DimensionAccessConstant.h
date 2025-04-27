#ifndef MARCO_MODELING_DIMENSIONACCESSCONSTANT_H
#define MARCO_MODELING_DIMENSIONACCESSCONSTANT_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling {
class DimensionAccessConstant : public DimensionAccess {
public:
  DimensionAccessConstant(mlir::MLIRContext *context, int64_t value);

  static bool classof(const DimensionAccess *obj) {
    return obj->getKind() == DimensionAccess::Kind::Constant;
  }

  [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

  [[nodiscard]] bool operator==(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator==(const DimensionAccessConstant &other) const;

  [[nodiscard]] bool operator!=(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator!=(const DimensionAccessConstant &other) const;

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

  [[nodiscard]] int64_t getValue() const;

private:
  int64_t value;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_DIMENSIONACCESSCONSTANT_H
