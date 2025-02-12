#ifndef MARCO_MODELING_DIMENSIONACCESSDIV_H
#define MARCO_MODELING_DIMENSIONACCESSDIV_H

#include "marco/Modeling/DimensionAccess.h"

namespace marco::modeling {
class DimensionAccessDiv : public DimensionAccess {
public:
  DimensionAccessDiv(mlir::MLIRContext *context,
                     std::unique_ptr<DimensionAccess> first,
                     std::unique_ptr<DimensionAccess> second);

  DimensionAccessDiv(const DimensionAccessDiv &other);

  DimensionAccessDiv(DimensionAccessDiv &&other) noexcept;

  ~DimensionAccessDiv() override;

  DimensionAccessDiv &operator=(const DimensionAccessDiv &other);

  DimensionAccessDiv &operator=(DimensionAccessDiv &&other) noexcept;

  friend void swap(DimensionAccessDiv &first, DimensionAccessDiv &second);

  static bool classof(const DimensionAccess *obj) {
    return obj->getKind() == DimensionAccess::Kind::Div;
  }

  [[nodiscard]] std::unique_ptr<DimensionAccess> clone() const override;

  [[nodiscard]] bool operator==(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator==(const DimensionAccessDiv &other) const;

  [[nodiscard]] bool operator!=(const DimensionAccess &other) const override;

  [[nodiscard]] bool operator!=(const DimensionAccessDiv &other) const;

  llvm::raw_ostream &dump(llvm::raw_ostream &os,
                          const llvm::DenseMap<const IndexSet *, uint64_t>
                              &iterationSpacesIds) const override;

  void collectIterationSpaces(
      llvm::DenseSet<const IndexSet *> &iterationSpaces) const override;

  void collectIterationSpaces(
      llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
      llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
          &dependentDimensions) const override;

  [[nodiscard]] bool isAffine() const override;

  [[nodiscard]] mlir::AffineExpr getAffineExpr() const override;

  [[nodiscard]] mlir::AffineExpr
  getAffineExpr(unsigned int numOfDimensions,
                FakeDimensionsMap &fakeDimensionsMap) const override;

  [[nodiscard]] DimensionAccess &getFirst();

  [[nodiscard]] const DimensionAccess &getFirst() const;

  [[nodiscard]] DimensionAccess &getSecond();

  [[nodiscard]] const DimensionAccess &getSecond() const;

  [[nodiscard]] IndexSet map(const Point &point,
                             llvm::DenseMap<const IndexSet *, Point>
                                 &currentIndexSetsPoint) const override;

private:
  std::unique_ptr<DimensionAccess> first;
  std::unique_ptr<DimensionAccess> second;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_DIMENSIONACCESSDIV_H
