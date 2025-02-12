#ifndef MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
#define MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H

#include "marco/Modeling/AccessFunctionAffineMap.h"

namespace marco::modeling {
class AccessFunctionRotoTranslation : public AccessFunctionAffineMap {
public:
  static bool
  canBeBuilt(unsigned int numOfDimensions,
             llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  static bool canBeBuilt(mlir::AffineMap affineMap);

  explicit AccessFunctionRotoTranslation(mlir::AffineMap affineMap);

  AccessFunctionRotoTranslation(
      mlir::MLIRContext *context, uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  ~AccessFunctionRotoTranslation() override;

  [[nodiscard]] std::unique_ptr<AccessFunction> clone() const override;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const AccessFunction *obj) {
    return obj->getKind() == RotoTranslation;
  }

  /// }

  [[nodiscard]] bool operator==(const AccessFunction &other) const override;

  [[nodiscard]] bool
  operator==(const AccessFunctionRotoTranslation &other) const;

  [[nodiscard]] bool operator!=(const AccessFunction &other) const override;

  [[nodiscard]] bool
  operator!=(const AccessFunctionRotoTranslation &other) const;

  [[nodiscard]] bool isInvertible() const override;

  [[nodiscard]] std::unique_ptr<AccessFunction> inverse() const override;

  using AccessFunction::map;

  [[nodiscard]] MultidimensionalRange
  map(const MultidimensionalRange &indices) const;

  [[nodiscard]] IndexSet map(const IndexSet &indices) const override;

  [[nodiscard]] IndexSet
  inverseMap(const IndexSet &accessedIndices,
             const IndexSet &parentIndices) const override;

  /// Check if each i-th dimension is accessed at the i-th position (with
  /// an optional offset).
  [[nodiscard]] bool isIdentityLike() const;

  void countVariablesUsages(llvm::SmallVectorImpl<size_t> &usages) const;

  [[nodiscard]] std::optional<uint64_t>
  getInductionVariableIndex(uint64_t expressionIndex) const;

  [[nodiscard]] int64_t getOffset(uint64_t expressionIndex) const;

  bool isScalarIndependent(const AccessFunction &other,
                           const IndexSet &sourceIndices) const override;

private:
  [[nodiscard]] std::optional<uint64_t>
  getInductionVariableIndex(mlir::AffineExpr expression) const;

  [[nodiscard]] int64_t getOffset(mlir::AffineExpr expression) const;

  bool isScalarIndependent(const AccessFunctionRotoTranslation &other,
                           const IndexSet &sourceIndices) const;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
