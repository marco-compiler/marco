#ifndef MARCO_CODEGEN_VARIABLE_H
#define MARCO_CODEGEN_VARIABLE_H

#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/modeling/Dependency.h"
#include "marco/modeling/Matching.h"
#include <memory>

namespace marco::codegen
{
  /// Proxy to the value representing a variable within the model.
  class Variable
  {
    public:
      class Impl;
      using Id = mlir::Operation*;

      static std::unique_ptr<Variable> build(mlir::Value value);

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

  /// Container for the variables of the model.
  /// It uses a shared_ptr in order to allow to easily share the same set of variables among different
  /// equations without the need of copying all of them each time.
  /// Each variable is also stored as a unique_ptr so that we can assume its address not to change due
  /// to the resizing of its container.
  class Variables
  {
    public:
      using Container = std::vector<std::unique_ptr<Variable>>;
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      Variables();

      void add(std::unique_ptr<Variable> variable);

      size_t size() const;

      std::unique_ptr<Variable>& operator[](size_t index);

      const std::unique_ptr<Variable>& operator[](size_t index) const;

      iterator begin();

      const_iterator begin() const;

      iterator end();

      const_iterator end() const;

    private:
      class Impl;
      std::shared_ptr<Impl> impl;
  };
}

// Traits specializations for the modeling library
namespace marco::modeling
{
  namespace dependency
  {
    template<>
    struct VariableTraits<::marco::codegen::Variable*>
    {
      using Variable = ::marco::codegen::Variable*;
      using Id = mlir::Operation*;

      static Id getId(const Variable* variable)
      {
        return (*variable)->getId();
      }
    };
  }

  namespace matching
  {
    template<>
    struct VariableTraits<::marco::codegen::Variable*>
    {
      using Variable = ::marco::codegen::Variable*;
      using Id = mlir::Operation*;

      static Id getId(const Variable* variable)
      {
        return (*variable)->getId();
      }

      static size_t getRank(const Variable* variable)
      {
        return (*variable)->getRank();
      }

      static long getDimensionSize(const Variable* variable, size_t index)
      {
        return (*variable)->getDimensionSize(index);
      }
    };
  }
}

#endif // MARCO_CODEGEN_VARIABLE_H
