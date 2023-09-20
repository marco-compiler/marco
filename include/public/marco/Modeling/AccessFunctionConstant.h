#ifndef MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H
#define MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionConstant : public AccessFunction
  {
    public:
      AccessFunctionConstant(mlir::AffineMap affineMap);

      ~AccessFunctionConstant();

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Constant;
      }

      /// }

      static bool canBeBuilt(mlir::AffineMap affineMap);

      std::unique_ptr<AccessFunction> clone() const override;

      using AccessFunction::map;

      IndexSet map(const IndexSet& indices) const override;

      IndexSet inverseMap(
          const IndexSet& accessedIndices,
          const IndexSet& parentIndices) const override;
  };
}

#endif // MARCO_MODELING_ACCESSFUNCTIONCONSTANT_H