#ifndef MARCO_MODELING_DIMENSIONACCESSINDICES_H
#define MARCO_MODELING_DIMENSIONACCESSINDICES_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/DenseSet.h"

namespace marco::modeling {
class DimensionAccessIndices : public DimensionAccess {
public:
  DimensionAccessIndices(mlir::MLIRContext *context,
                         std::shared_ptr<IndexSet> space, uint64_t dimension,
                         llvm::DenseSet<uint64_t> dimensionDependencies);

  static bool classof(const DimensionAccess *obj) {
    return obj->getKind() == DimensionAccess::Kind::Indices;
  }

  [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

  [[nodiscard]] bool operator==(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator==(const DimensionAccessIndices &other) const;

  [[nodiscard]] bool operator!=(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator!=(const DimensionAccessIndices &other) const;

  llvm::raw_ostream &dump(llvm::raw_ostream &os,
                          const llvm::DenseMap<const IndexSet *, uint64_t>
                              &iterationSpacesIds) const override;

  void collectIterationSpaces(
      llvm::DenseSet<const IndexSet *> &iterationSpaces) const override;

  void collectIterationSpaces(
      llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
      llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
          &dependentDimensions) const override;

  [[nodiscard]] bool isConstant() const override;

  [[nodiscard]] mlir::AffineExpr
  getAffineExpr(unsigned int numOfDimensions,
                FakeDimensionsMap &fakeDimensionsMap) const override;

  [[nodiscard]] IndexSet map(const Point &point,
                             llvm::DenseMap<const IndexSet *, Point>
                                 &currentIndexSetsPoint) const override;

  [[nodiscard]] IndexSet &getIndices();

  [[nodiscard]] const IndexSet &getIndices() const;

private:
  std::shared_ptr<IndexSet> space;
  uint64_t dimension;
  llvm::DenseSet<uint64_t> dimensionDependencies;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_DIMENSIONACCESSINDICES_H
