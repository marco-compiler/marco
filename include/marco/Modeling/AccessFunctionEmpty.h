#ifndef MARCO_MODELING_ACCESSFUNCTIONEMPTY_H
#define MARCO_MODELING_ACCESSFUNCTIONEMPTY_H

#include "marco/Modeling/AccessFunctionAffineMap.h"

namespace marco::modeling {
class AccessFunctionEmpty : public AccessFunctionAffineMap {
public:
  static bool
  canBeBuilt(uint64_t numOfDimensions,
             llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  static bool canBeBuilt(mlir::AffineMap affineMap);

  explicit AccessFunctionEmpty(mlir::AffineMap affineMap);

  AccessFunctionEmpty(mlir::MLIRContext *context, uint64_t numOfDimensions,
                      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  ~AccessFunctionEmpty() override;

  [[nodiscard]] std::unique_ptr<AccessFunction> clone() const override;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const AccessFunction *obj) {
    return obj->getKind() == Kind::Empty;
  }

  /// }

  [[nodiscard]] bool operator==(const AccessFunction &other) const override;

  [[nodiscard]] bool operator==(const AccessFunctionEmpty &other) const;

  [[nodiscard]] bool operator!=(const AccessFunction &other) const override;

  [[nodiscard]] bool operator!=(const AccessFunctionEmpty &other) const;

  [[nodiscard]] bool isConstant() const override;

  [[nodiscard]] bool isInvertible() const override;

  [[nodiscard]] std::unique_ptr<AccessFunction> inverse() const override;

  [[nodiscard]] IndexSet map(const Point &point) const override;

  [[nodiscard]] IndexSet map(const IndexSet &indices) const override;

  [[nodiscard]] IndexSet
  inverseMap(const IndexSet &accessedIndices,
             const IndexSet &parentIndices) const override;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTIONEMPTY_H
