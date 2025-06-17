#ifndef MARCO_MODELING_ACCESSFUNCTIONAFFINECONSTANT_H
#define MARCO_MODELING_ACCESSFUNCTIONAFFINECONSTANT_H

#include "marco/Modeling/AccessFunctionAffineMap.h"

namespace marco::modeling {
class AccessFunctionAffineConstant : public AccessFunctionAffineMap {
public:
  static bool
  canBeBuilt(llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  static bool canBeBuilt(mlir::AffineMap affineMap);

  explicit AccessFunctionAffineConstant(mlir::AffineMap affineMap);

  AccessFunctionAffineConstant(
      mlir::MLIRContext *context, uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  ~AccessFunctionAffineConstant() override;

  [[nodiscard]] std::unique_ptr<AccessFunction> clone() const override;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const AccessFunction *obj) {
    return obj->getKind() == Kind::Affine_Constant;
  }

  /// }

  [[nodiscard]] bool operator==(const AccessFunction &other) const override;

  [[nodiscard]] bool
  operator==(const AccessFunctionAffineConstant &other) const;

  [[nodiscard]] bool operator!=(const AccessFunction &other) const override;

  [[nodiscard]] bool
  operator!=(const AccessFunctionAffineConstant &other) const;

  [[nodiscard]] bool isConstant() const override;

  using AccessFunctionAffineMap::map;

  [[nodiscard]] IndexSet map(const IndexSet &indices) const override;

  [[nodiscard]] IndexSet
  inverseMap(const IndexSet &accessedIndices,
             const IndexSet &parentIndices) const override;

  [[nodiscard]] std::optional<Point> getMappedPoint() const;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTIONAFFINECONSTANT_H
