#ifndef MARCO_CODEGEN_VARIABLE_H
#define MARCO_CODEGEN_VARIABLE_H

#include <marco/codegen/dialects/modelica/ModelicaDialect.h>
#include <memory>

namespace marco::codegen
{
  /**
   * Proxy to the value representing a variable within the model.
   */
  class Variable
  {
    public:
    class Impl;
    using Id = mlir::Operation*;

    Variable(mlir::Value value);

    Variable(const Variable& other);

    ~Variable();

    Variable& operator=(const Variable& other);
    Variable& operator=(Variable&& other);

    friend void swap(Variable& first, Variable& second);

    Id getId() const;
    size_t getRank() const;
    long getDimensionSize(size_t index) const;

    mlir::Value getValue() const;
    modelica::MemberCreateOp getDefiningOp() const;
    bool isConstant() const;

    private:
    std::unique_ptr<Impl> impl;
  };
}

#endif // MARCO_CODEGEN_VARIABLE_H
