#ifndef MARCO_CODEGEN_TRANSFORMS_MODELING_VARIABLEBRIDGE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELING_VARIABLEBRIDGE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/Matching.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::modelica::bridge
{
  class VariableBridge
  {
    public:
      mlir::SymbolRefAttr name;
      marco::modeling::IndexSet indices;

    public:
      static std::unique_ptr<VariableBridge> build(
          mlir::SymbolRefAttr name,
          IndexSet indices);

      static std::unique_ptr<VariableBridge> build(VariableOp variable);

      VariableBridge(
          mlir::SymbolRefAttr name,
          marco::modeling::IndexSet indices);

      // Forbid copies to avoid dangling pointers by design.
      VariableBridge(const VariableBridge& other) = delete;
      VariableBridge(VariableBridge&& other) = delete;
      VariableBridge& operator=(const VariableBridge& other) = delete;
      VariableBridge& operator==(const VariableBridge& other) = delete;
  };
}

namespace marco::modeling::matching
{
  template<>
  struct VariableTraits<::mlir::modelica::bridge::VariableBridge*>
  {
    using Variable = ::mlir::modelica::bridge::VariableBridge*;
    using Id = ::mlir::modelica::bridge::VariableBridge*;

    static Id getId(const Variable* variable)
    {
      return *variable;
    }

    static size_t getRank(const Variable* variable)
    {
      size_t rank = (*variable)->indices.rank();

      if (rank == 0) {
        return 1;
      }

      return rank;
    }

    static IndexSet getIndices(const Variable* variable)
    {
      const IndexSet& result = (*variable)->indices;

      if (result.empty()) {
        return {Point(0)};
      }

      return result;
    }
  };
}

namespace marco::modeling::dependency
{
  template<>
  struct VariableTraits<::mlir::modelica::bridge::VariableBridge*>
  {
    using Variable = ::mlir::modelica::bridge::VariableBridge*;
    using Id = ::mlir::modelica::bridge::VariableBridge*;

    static Id getId(const Variable* variable)
    {
      return *variable;
    }

    static size_t getRank(const Variable* variable)
    {
      size_t rank = (*variable)->indices.rank();

      if (rank == 0) {
        return 1;
      }

      return rank;
    }

    static IndexSet getIndices(const Variable* variable)
    {
      const IndexSet& result = (*variable)->indices;

      if (result.empty()) {
        return {Point(0)};
      }

      return result;
    }
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_VARIABLEBRIDGE_H
