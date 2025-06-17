#ifndef MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H
#define MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H

#include "marco/Modeling/AccessFunctionGeneric.h"

namespace marco::modeling {
class AccessFunctionConstant : public AccessFunctionGeneric {
public:
  static bool
  canBeBuilt(llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  static bool canBeBuilt(mlir::AffineMap affineMap);

  AccessFunctionConstant(
      mlir::MLIRContext *context, uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  explicit AccessFunctionConstant(mlir::AffineMap affineMap);

  ~AccessFunctionConstant() override;

  [[nodiscard]] std::unique_ptr<AccessFunction> clone() const override;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const AccessFunction *obj) {
    return obj->getKind() == Kind::Constant;
  }

  /// }

  [[nodiscard]] bool isConstant() const override;

  using AccessFunctionGeneric::map;

  [[nodiscard]] IndexSet map(const IndexSet &indices) const override;

  [[nodiscard]] IndexSet
  inverseMap(const IndexSet &accessedIndices,
             const IndexSet &parentIndices) const override;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H
