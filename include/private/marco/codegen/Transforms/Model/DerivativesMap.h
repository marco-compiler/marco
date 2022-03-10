#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_DERIVATIVESMAP_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_DERIVATIVESMAP_H

#include "llvm/ADT/STLExtras.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Value.h"
#include <map>

namespace marco::codegen
{
  namespace detail
  {
    class Derivative
    {
      public:
        Derivative(mlir::Value variable, mlir::Value derivative)
          : variable(std::move(variable)), derivative(std::move(derivative))
        {
        }

        mlir::Value getDerivedVariable() const
        {
          return variable;
        }

        marco::modeling::IndexSet getDerivedIndices() const
        {
          return indices;
        }

        mlir::Value getDerivative() const
        {
          return derivative;
        }

      private:
        mlir::Value variable;
        mlir::Value derivative;
        marco::modeling::IndexSet indices;
    };

    /// Comparator for mlir::Value.
    struct ValueComparator
    {
      bool operator()(const mlir::Value& first, const mlir::Value& second) const
      {
        if (first.isa<mlir::BlockArgument>() && second.isa<mlir::BlockArgument>()) {
          return first.cast<mlir::BlockArgument>().getArgNumber() < second.cast<mlir::BlockArgument>().getArgNumber();
        }

        mlir::Operation* op1 = first.getDefiningOp();
        mlir::Operation* op2 = second.getDefiningOp();

        if (op1 != nullptr && op2 != nullptr) {
          return op1 < op2;
        }

        return true;
      }
    };
  }

  class DerivativesMap
  {
    public:
      bool hasDerivative(mlir::Value variable) const
      {
        return derivatives.find(variable) != derivatives.end();
      }

      modeling::IndexSet getDerivedIndices(mlir::Value variable) const
      {
        modeling::IndexSet result;

        for (const auto& derivative : llvm::make_range(derivatives.equal_range(variable))) {
          result += derivative.second.getDerivedIndices();
        }

        return result;
      }

    private:
      std::multimap<mlir::Value, detail::Derivative, detail::ValueComparator> derivatives;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_DERIVATIVESMAP_H
