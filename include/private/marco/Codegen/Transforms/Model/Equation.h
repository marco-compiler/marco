#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_EQUATION_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_EQUATION_H

#include "llvm/Support/raw_ostream.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/Model/Access.h"
#include "marco/Codegen/Transforms/Model/Path.h"
#include "marco/Codegen/Transforms/Model/Variable.h"
#include "marco/Modeling/Matching.h"
#include "marco/Modeling/Scheduling.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class Equation
  {
    public:
      static std::unique_ptr<Equation> build(
          mlir::modelica::EquationOp equation, Variables variables);

      virtual ~Equation();

      virtual std::unique_ptr<Equation> clone() const = 0;

      virtual mlir::modelica::EquationOp cloneIR() const = 0;

      virtual void eraseIR() = 0;

      virtual void dumpIR() const;

      virtual void dumpIR(llvm::raw_ostream& os) const = 0;

      /// Get the IR operation.
      virtual mlir::modelica::EquationOp getOperation() const = 0;

      /// Get the variables considered by the equation while determining the accesses.
      virtual Variables getVariables() const = 0;

      /// Set the variables considered by the equation while determining the accesses.
      virtual void setVariables(Variables variables) = 0;

      /// Get the number of induction variables.
      virtual size_t getNumOfIterationVars() const = 0;

      /// Get the iteration ranges.
      virtual modeling::MultidimensionalRange getIterationRanges() const = 0;

      /// Get the accesses to variables.
      virtual std::vector<Access> getAccesses() const = 0;

      virtual ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const = 0;

      /// Get the IR value at a given path.
      virtual mlir::Value getValueAtPath(const EquationPath& path) const = 0;

      /// Transform the equation IR such that the access at the given equation path is the
      /// only term on the left hand side of the equation.
      virtual mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder, const EquationPath& path) = 0;

      /// Clone the equation IR and make it explicit with respect to the given equation path.
      virtual std::unique_ptr<Equation> cloneIRAndExplicitate(
          mlir::OpBuilder& builder, const EquationPath& path) const = 0;

      virtual std::vector<mlir::Value> getInductionVariables() const = 0;

      /// Replace an access in an equation with the right-hand side of the current one.
      /// This equation is assumed to already be explicit.
      virtual mlir::LogicalResult replaceInto(
          mlir::OpBuilder& builder,
          Equation& destination,
          const ::marco::modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath) const = 0;

      virtual mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          ::marco::modeling::scheduling::Direction iterationDirection) const = 0;

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

      /// Obtain the access to a variable that is reached through a given path inside the equation.
      Access getAccessFromPath(const EquationPath& path) const;

      std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const;
  };

  /// Mark an equation as temporary and delete the underlying IR when the guard goes out of scope.
  class TemporaryEquationGuard
  {
    public:
      TemporaryEquationGuard(Equation& equation);

      TemporaryEquationGuard(const TemporaryEquationGuard& other) = delete;

      ~TemporaryEquationGuard();

    private:
      Equation* equation;
  };

  namespace impl
  {
    // This class must be specialized for the type that is used as container of the equations.
    template<
        typename EquationType,
        template<typename... Args> class Container>
    struct EquationsTrait
    {
      // Elements to provide:
      //
      // static void add(Container<std::unique_ptr<EquationType>>& container, std::unique_ptr<EquationType> equation);
      // static size_t size(const Container<std::unique_ptr<EquationType>>& container);
      // static std::unique_ptr<EquationType>& get(Container<std::unique_ptr<EquationType>>& container, size_t index);
      // static const std::unique_ptr<EquationType>& get(const Container<std::unique_ptr<EquationType>>& container, size_t index);
    };

    /// Implementation of the equations container.
    /// The actual container can be specified by means of the template parameter.
    template<
        typename EquationType,
        template<typename... Args> class Container>
    class BaseEquations
    {
      public:
        using iterator = typename Container<std::unique_ptr<EquationType>>::iterator;
        using const_iterator = typename Container<std::unique_ptr<EquationType>>::const_iterator;

      private:
        using Trait = EquationsTrait<EquationType, std::vector>;

      public:
        void add(std::unique_ptr<EquationType> equation)
        {
          Trait::add(equations, std::move(equation));
        }

        size_t size() const
        {
          return Trait::size(equations);
        }

        std::unique_ptr<EquationType>& operator[](size_t index)
        {
          assert(index < size());
          return Trait::get(equations, index);
        }

        const std::unique_ptr<EquationType>& operator[](size_t index) const
        {
          assert(index < size());
          return Trait::get(equations, index);
        }

        BaseEquations::iterator begin()
        {
          using std::begin;
          return begin(equations);
        }

        BaseEquations::const_iterator begin() const
        {
          using std::begin;
          return begin(equations);
        }

        BaseEquations::iterator end()
        {
          using std::end;
          return end(equations);
        }

        BaseEquations::const_iterator end() const
        {
          using std::end;
          return end(equations);
        }

        /// For each equation, set the variables it should consider while determining the accesses.
        void setVariables(Variables variables)
        {
          for (auto& equation : *this) {
            equation->setVariables(variables);
          }
        }

      private:
        Container<std::unique_ptr<EquationType>> equations;
    };

    template<
        typename EquationType,
        template<typename... Args> class Container = std::vector>
    class Equations : public BaseEquations<EquationType, Container>
    {
      public:
        using iterator = typename BaseEquations<EquationType, std::vector>::iterator;
        using const_iterator = typename BaseEquations<EquationType, std::vector>::const_iterator;
    };

    /// Specialization of the equations container with std::vector as container.
    template<typename EquationType>
    class Equations<EquationType, std::vector> : public BaseEquations<EquationType, std::vector>
    {
      public:
        using iterator = typename BaseEquations<EquationType, std::vector>::iterator;
        using const_iterator = typename BaseEquations<EquationType, std::vector>::const_iterator;
    };

    template<typename EquationType>
    class EquationsTrait<EquationType, std::vector>
    {
      public:
        static void add(
            std::vector<std::unique_ptr<EquationType>>& container,
            std::unique_ptr<EquationType> equation)
        {
          container.push_back(std::move(equation));
        }

        static size_t size(const std::vector<std::unique_ptr<EquationType>>& container)
        {
          return container.size();
        }

        static std::unique_ptr<EquationType>& get(
            std::vector<std::unique_ptr<EquationType>>& container, size_t index)
        {
          return container[index];
        }

        static const std::unique_ptr<EquationType>& get(
            const std::vector<std::unique_ptr<EquationType>>& container, size_t index)
        {
          return container[index];
        }
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

      /// @name Forwarded methods
      /// {

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

      void setVariables(Variables variables)
      {
        impl->setVariables(std::move(variables));
      }

      /// }

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

    static MultidimensionalRange getIterationRanges(const Equation* equation)
    {
      return (*equation)->getIterationRanges();
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

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_EQUATION_H
