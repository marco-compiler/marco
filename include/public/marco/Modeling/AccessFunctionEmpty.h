#ifndef MARCO_MODELING_ACCESSFUNCTIONEMPTY_H
#define MARCO_MODELING_ACCESSFUNCTIONEMPTY_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionEmpty : public AccessFunction
  {
    public:
      AccessFunctionEmpty(mlir::AffineMap affineMap);

      ~AccessFunctionEmpty();

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Empty;
      }

      /// }

      static bool canBeBuilt(mlir::AffineMap affineMap);

      std::unique_ptr<AccessFunction> clone() const override;

      bool isInvertible() const override;

      std::unique_ptr<AccessFunction> inverse() const override;

      using AccessFunction::map;

      IndexSet map(const IndexSet& indices) const override;
  };
}

#endif // MARCO_MODELING_ACCESSFUNCTIONEMPTY_H
