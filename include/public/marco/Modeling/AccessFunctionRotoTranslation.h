#ifndef MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
#define MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionRotoTranslation : public AccessFunction
  {
    public:
      AccessFunctionRotoTranslation(mlir::AffineMap affineMap);

      ~AccessFunctionRotoTranslation();

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == RotoTranslation;
      }

      /// }

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

      std::optional<unsigned int>
      getInductionVariableIndex(unsigned int expressionIndex) const;

      int64_t getOffset(unsigned int expressionIndex) const;

    private:
      std::optional<unsigned int> getInductionVariableIndex(
          mlir::AffineExpr expression) const;

      int64_t getOffset(mlir::AffineExpr expression) const;
  };
}

#endif // MARCO_MODELING_ACCESSFUNCTIONROTOTRANSLATION_H
