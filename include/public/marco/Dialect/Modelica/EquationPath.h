#ifndef MARCO_DIALECT_MODELICA_EQUATIONPATH_H
#define MARCO_DIALECT_MODELICA_EQUATIONPATH_H

#include "marco/Dialect/Modelica/ExpressionPath.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::modelica
{
  class EquationPath
  {
    public:
      using const_iterator = ExpressionPath::const_iterator;

      enum EquationSide
      {
        LEFT,
        RIGHT
      };

      class Guard : public ExpressionPath::Guard
      {
        public:
          Guard(EquationPath& path);
      };

      EquationPath(
          EquationSide equationSide,
          llvm::ArrayRef<uint64_t> path = {});

      EquationPath(EquationSide equationSide, ExpressionPath path);

      friend llvm::hash_code hash_value(const EquationPath& value);

      bool operator==(const EquationPath& other) const;
      bool operator!=(const EquationPath& other) const;

      EquationSide getEquationSide() const;

      uint64_t operator[](size_t index) const;

      const_iterator begin() const;
      const_iterator end() const;

      EquationPath operator+(uint64_t index) const;

      EquationPath& operator+=(uint64_t index);

      size_t size() const;

    private:
      EquationSide equationSide;
      ExpressionPath expressionPath;
  };
}

#endif // MARCO_DIALECT_MODELICA_EQUATIONPATH_H
