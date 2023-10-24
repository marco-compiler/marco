#ifndef MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
#define MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionRotoTranslation : public AccessFunction
  {
    public:
      AccessFunctionRotoTranslation(
          mlir::MLIRContext* context,
          unsigned int numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

      explicit AccessFunctionRotoTranslation(mlir::AffineMap affineMap);

      ~AccessFunctionRotoTranslation() override;

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == RotoTranslation;
      }

      /// }

      static bool canBeBuilt(
          unsigned int numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

      static bool canBeBuilt(mlir::AffineMap affineMap);

      std::unique_ptr<AccessFunction> clone() const override;

      bool isInvertible() const override;

      std::unique_ptr<AccessFunction> inverse() const override;

      using AccessFunction::map;

      MultidimensionalRange map(const MultidimensionalRange& indices) const;

      IndexSet map(const IndexSet& indices) const override;

      IndexSet inverseMap(
          const IndexSet& accessedIndices,
          const IndexSet& parentIndices) const override;

      /// Check if each i-th dimension is accessed at the i-th position (with
      /// an optional offset).
      bool isIdentityLike() const;

      void countVariablesUsages(llvm::SmallVectorImpl<size_t>& usages) const;

      std::optional<uint64_t>
      getInductionVariableIndex(unsigned int expressionIndex) const;

      int64_t getOffset(unsigned int expressionIndex) const;
  };
}

#endif // MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
