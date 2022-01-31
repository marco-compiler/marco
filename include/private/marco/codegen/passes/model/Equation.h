#ifndef MARCO_CODEGEN_EQUATION_H
#define MARCO_CODEGEN_EQUATION_H

#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Access.h"
#include "marco/codegen/passes/model/Path.h"
#include "marco/codegen/passes/model/Variable.h"
#include "marco/modeling/Matching.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace marco::codegen
{
  class Equation
  {
    public:
      static std::unique_ptr<Equation> build(
          modelica::EquationOp equation, Variables variables);

      virtual std::unique_ptr<Equation> clone() const = 0;

      virtual modelica::EquationOp cloneIR() const = 0;

      /// Get the IR operation.
      virtual modelica::EquationOp getOperation() const = 0;

      /// Get the variables considered by the equation while determining the accesses.
      virtual const Variables& getVariables() const = 0;

      /// Set the variables considered by the equation while determining the accesses.
      virtual void setVariables(Variables variables) = 0;

      virtual size_t getNumOfIterationVars() const = 0;
      virtual long getRangeBegin(size_t inductionVarIndex) const = 0;
      virtual long getRangeEnd(size_t inductionVarIndex) const = 0;

      virtual std::vector<Access> getAccesses() const = 0;

      virtual ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const = 0;

      virtual mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars) const = 0;

    protected:
      llvm::Optional<Variable*> findVariable(mlir::Value value) const;

      bool isVariable(mlir::Value value) const;

      bool isReferenceAccess(mlir::Value value) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Value value,
          EquationPath path) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Value value,
          std::vector<::marco::modeling::DimensionAccess>& dimensionAccesses,
          EquationPath path) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Operation* op,
          std::vector<::marco::modeling::DimensionAccess>& dimensionAccesses,
          EquationPath path) const;

      void resolveAccess(
          std::vector<Access>& accesses,
          mlir::Value value,
          std::vector<::marco::modeling::DimensionAccess>& dimensionsAccesses,
          EquationPath path) const;

      Access getAccessFromPath(const EquationPath& path) const;

      std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const;
  };

  namespace impl
  {
    /// Implementation of the equations container.
    template<typename EquationType>
    class Equations
    {
      public:
        using Container = std::vector<std::unique_ptr<EquationType>>;
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;

        void add(std::unique_ptr<EquationType> equation)
        {
          equations.push_back(std::move(equation));
        }

        size_t size() const
        {
          return equations.size();
        }

        std::unique_ptr<EquationType>& operator[](size_t index)
        {
          assert(index < size());
          return equations[index];
        }

        const std::unique_ptr<EquationType>& operator[](size_t index) const
        {
          assert(index < size());
          return equations[index];
        }

        Equations::iterator begin()
        {
          return equations.begin();
        }

        Equations::const_iterator begin() const
        {
          return equations.begin();
        }

        Equations::iterator end()
        {
          return equations.end();
        }

        Equations::const_iterator end() const
        {
          return equations.end();
        }

      private:
        Container equations;
    };
  }

  /// Container for the equations of the model.
  /// The container has value semantics. In fact, the implementation consists in a shared pointer and a copy
  /// of the container would refer to the same set of equations.
  /// The template parameter is used to control which type of equations it should contain.
  template<typename EquationType = Equation>
  class Equations
  {
    private:
      using Impl = impl::Equations<EquationType>;

    public:
      using iterator = typename Impl::iterator;
      using const_iterator = typename Impl::const_iterator;

      Equations() : impl(std::make_shared<Impl>())
      {
      }

      void add(std::unique_ptr<EquationType> equation)
      {
        impl->add(std::move(equation));
      }

      size_t size() const
      {
        return impl->size();
      }

      std::unique_ptr<EquationType>& operator[](size_t index)
      {
        return (*impl)[index];
      }

      const std::unique_ptr<EquationType>& operator[](size_t index) const
      {
        return (*impl)[index];
      }

      iterator begin()
      {
        return impl->begin();
      }

      const_iterator begin() const
      {
        return impl->begin();
      }

      iterator end()
      {
        return impl->end();
      }

      const_iterator end() const
      {
        return impl->end();
      }

      /// For each equation, set the variables it should consider while determining the accesses.
      void setVariables(Variables variables)
      {
        for (auto& equation : *this) {
          equation->setVariables(variables);
        }
      }

    private:
      std::shared_ptr<Impl> impl;
  };
}

// Traits specializations for the modeling library
namespace marco::modeling::matching
{
  template<>
  struct EquationTraits<::marco::codegen::Equation*>
  {
    using Equation = ::marco::codegen::Equation*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation)
    {
      return (*equation)->getOperation().getOperation();
    }

    static size_t getNumOfIterationVars(const Equation* equation)
    {
      return (*equation)->getNumOfIterationVars();
    }

    static long getRangeBegin(const Equation* equation, size_t inductionVarIndex)
    {
      return (*equation)->getRangeBegin(inductionVarIndex);
    }

    static long getRangeEnd(const Equation* equation, size_t inductionVarIndex)
    {
      return (*equation)->getRangeEnd(inductionVarIndex);
    }

    using VariableType = codegen::Variable*;

    using AccessProperty = codegen::EquationPath;

    static std::vector<Access<VariableType, AccessProperty>> getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> accesses;

      for (const auto& access : (*equation)->getAccesses()) {
        accesses.emplace_back(access.getVariable(), access.getAccessFunction(), access.getPath());
      }

      return accesses;
    }
  };
}

#endif // MARCO_CODEGEN_EQUATION_H
