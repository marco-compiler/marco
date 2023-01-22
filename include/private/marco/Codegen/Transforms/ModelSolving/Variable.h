#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/Matching.h"
#include <memory>

namespace marco::codegen
{
  /// Proxy to the value representing a variable within the model.
  class Variable
  {
    public:
      using Id = mlir::Operation*;

      static std::unique_ptr<Variable> build(mlir::Value value);

      virtual ~Variable();

      virtual std::unique_ptr<Variable> clone() const = 0;

      bool operator==(mlir::Value value) const;

      virtual Id getId() const = 0;
      virtual size_t getRank() const = 0;
      virtual long getDimensionSize(size_t index) const = 0;

      virtual modeling::IndexSet getIndices() const = 0;

      virtual mlir::Value getValue() const = 0;
      virtual mlir::modelica::MemberCreateOp getDefiningOp() const = 0;
      virtual bool isParameter() const = 0;
  };

  /// Container for the variables of the model.
  /// It uses a shared_ptr in order to allow to easily share the same set of
  /// variables among different equations without the need of copying all of
  /// them each time.
  /// Each variable is also stored as a unique_ptr so that we can assume its
  /// address not to change due to the resizing of its container.
  class Variables
  {
    public:
      using Container = std::vector<std::unique_ptr<Variable>>;
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      Variables();

      size_t size() const;

      const std::unique_ptr<Variable>& operator[](size_t index) const;

      void add(std::unique_ptr<Variable> variable);

      iterator begin();

      const_iterator begin() const;

      iterator end();

      const_iterator end() const;

      bool isVariable(mlir::Value value) const;

      bool isReferenceAccess(mlir::Value value) const;

      llvm::Optional<Variable*> findVariable(mlir::Value value) const;

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

      static IndexSet getIndices(const Variable* variable)
      {
        return (*variable)->getIndices();
      }
    };
  }
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLE_H
