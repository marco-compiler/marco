#ifndef MARCO_MODELING_ACCESSFUNCTIONZERORESULTS_H
#define MARCO_MODELING_ACCESSFUNCTIONZERORESULTS_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionZeroResults : public AccessFunction
  {
    public:
      AccessFunctionZeroResults(mlir::AffineMap affineMap);

      ~AccessFunctionZeroResults();

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Empty;
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

#endif // MARCO_MODELING_ACCESSFUNCTIONZERORESULTS_H