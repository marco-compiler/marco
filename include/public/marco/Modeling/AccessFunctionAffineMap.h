#ifndef MARCO_MODELING_ACCESSFUNCTIONAFFINEMAP_H
#define MARCO_MODELING_ACCESSFUNCTIONAFFINEMAP_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling {
class AccessFunctionAffineMap : public AccessFunction {
public:
  static mlir::AffineMap
  buildAffineMap(mlir::MLIRContext *context, uint64_t numOfDimensions,
                 llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  AccessFunctionAffineMap(Kind kind, mlir::AffineMap affineMap);

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const AccessFunction *obj) {
    return obj->getKind() >= Kind::Affine &&
           obj->getKind() <= Kind::Affine_LastArgument;
  }

  /// }

  llvm::raw_ostream &dump(llvm::raw_ostream &os) const override;

  [[nodiscard]] uint64_t getNumOfDims() const override;

  [[nodiscard]] uint64_t getNumOfResults() const override;

  [[nodiscard]] bool isAffine() const override;

  [[nodiscard]] mlir::AffineMap getAffineMap() const override;

  [[nodiscard]] bool isIdentity() const override;

  [[nodiscard]] IndexSet map(const Point &point) const override;

  [[nodiscard]] IndexSet map(const IndexSet &indices) const override;

  [[nodiscard]] std::unique_ptr<AccessFunction>
  getWithGivenDimensions(uint64_t requestedDims) const override;

  [[nodiscard]] llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
  getGeneralizedAccesses() const override;

  [[nodiscard]] mlir::AffineMap getExtendedAffineMap(
      DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const override;

protected:
  [[nodiscard]] mlir::AffineMap
  getAffineMapWithGivenDimensions(uint64_t requestedDims) const;

private:
  mlir::AffineMap affineMap;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTIONAFFINEMAP_H
