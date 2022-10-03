#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATION_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Access.h"
#include "marco/Codegen/Transforms/ModelSolving/Path.h"
#include "marco/Codegen/Transforms/ModelSolving/Variable.h"
#include "marco/Modeling/Matching.h"
#include "marco/Modeling/Scheduling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class Equation
  {
    public:
      static std::unique_ptr<Equation> build(
          mlir::modelica::EquationInterface equation, Variables variables);

      virtual ~Equation();

      virtual std::unique_ptr<Equation> clone() const = 0;

      virtual mlir::modelica::EquationInterface cloneIR() const = 0;

      virtual void eraseIR() = 0;

      virtual void dumpIR() const;

      virtual void dumpIR(llvm::raw_ostream& os) const = 0;

      /// Get the IR operation.
      virtual mlir::modelica::EquationInterface getOperation() const = 0;

      /// Get the variables considered by the equation while determining the accesses.
      virtual Variables getVariables() const = 0;

      /// Set the variables considered by the equation while determining the accesses.
      virtual void setVariables(Variables variables) = 0;

      /// Get the number of induction variables.
      virtual size_t getNumOfIterationVars() const = 0;

      /// Get the iteration ranges.
      virtual modeling::IndexSet getIterationRanges() const = 0;

      /// Get the accesses to variables.
      virtual std::vector<Access> getAccesses() const = 0;

      virtual ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const = 0;

      /// Get the IR value at a given path.
      virtual mlir::Value getValueAtPath(const EquationPath& path) const = 0;

      /// Obtain the access to a variable that is reached through a given path inside the equation.
      virtual Access getAccessAtPath(const EquationPath& path) const = 0;

      /// Traverse an equation path.
      /// The function takes the path to be walked and a callback function
      /// that is called at each step with the current value. It must return
      /// 'true' to continue the visit, or 'false' to stop.
      virtual void traversePath(
          const EquationPath& path,
          std::function<bool(mlir::Value)> traverseFn) const = 0;

      /// Transform the equation IR such that the access at the given equation path is the
      /// only term on the left hand side of the equation.
      virtual mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          const EquationPath& path) = 0;

      /// Clone the equation IR and make it explicit with respect to the given equation path.
      virtual std::unique_ptr<Equation> cloneIRAndExplicitate(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          const EquationPath& path) const = 0;

      virtual std::vector<mlir::Value> getInductionVariables() const = 0;

      /// Replace an access in an equation with the right-hand side of the current one.
      /// This equation is assumed to already be explicit.
      virtual mlir::LogicalResult replaceInto(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          Equation& destination,
          const ::marco::modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath) const = 0;

      virtual mlir::func::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          ::marco::modeling::scheduling::Direction iterationDirection) const = 0;

      /// Populate the coefficients array and the constant term.
      /// \param builder The builder.
      /// \param coefficients The array of coefficients to be populated.
      /// \param constantTerm The constant term to be computed.
      /// \param equationIndices The indices of the equation
      /// \return Whether the values were computed successfully or not.
      virtual mlir::LogicalResult getCoefficients(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& coefficients,
          mlir::Value& constantTerm,
          const modeling::IndexSet& equationIndices) const = 0;

      /// Starting from a previously computed array of coefficients and a constant
      /// term, this function computes the coefficients to the equation variables
      /// for one side of the equation.
      /// \param builder The builder.
      /// \param coefficients The coefficients' starting values
      /// \param constantTerm The constant term's starting value
      /// \param values Array containing the values that summed make up one side of
      ///               the equation.
      /// \param side The side of the equation being considered.
      /// \param equationIndices The indices of the equation
      /// \return Whether the coefficient extraction is successful or not.
      virtual mlir::LogicalResult getSideCoefficients(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& coefficients,
          mlir::Value& constantTerm,
          std::vector<mlir::Value> values,
          EquationPath::EquationSide side,
          const modeling::IndexSet& equationIndices) const = 0;

      /// Take one side of the equation and convert it to a sum of terms. Collect
      /// those terms inside the output array.
      /// \param builder The builder.
      /// \param output The array that will contain the terms that summed up
      ///               constitutes the specified equation side.
      /// \param side The equation side to be considered.
      /// \return Whether the value collection was successful or not.
      virtual mlir::LogicalResult convertAndCollectSide(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& output,
          EquationPath::EquationSide side) const = 0;

      /// Replace the left and right hand sides of the equation with the given
      /// values.
      /// \param builder
      /// \param lhs Left hand side
      /// \param rhs Right hand side
      virtual void replaceSides(
          mlir::OpBuilder& builder,
          mlir::Value lhs,
          mlir::Value rhs) const = 0;

      /// Returns the flattened size of the system matrix until the specified
      /// variable, summing all the flat sizes of the variables preceding it.
      /// \param index The index of the variable to stop the sum
      /// \return The flattened size of the system matrix until the variable
      /// corresponding to the specified index.
      virtual size_t getSizeUntilVariable(
          size_t index) const = 0;

      /// Get the flattened access to the variable. This is used to get a unique
      /// identifier for an access to a non scalar variable. The rangeSet contains
      /// the information about the structure of the variable.
      /// \param access The access to the variable.
      /// \param equationIndices The indices of the equation.
      /// \param variableIndices This value contains the information about the structure
      ///                 of the multidimensional variable.
      /// \return The unique identifier for the input access.
      virtual size_t getFlatAccessIndex(
          const Access& access,
          const ::marco::modeling::IndexSet& equationIndices,
          const ::marco::modeling::IndexSet& variableIndices) const = 0;

    protected:
      llvm::Optional<Variable*> findVariable(mlir::Value value) const;

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

      std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const;

      mlir::LogicalResult checkLinearity(mlir::Value value) const;

      double getDoubleFromConstantValue(mlir::Value value) const;
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

    static IndexSet getIterationRanges(const Equation* equation)
    {
      return (*equation)->getIterationRanges();
    }

    using VariableType = codegen::Variable*;

    using AccessProperty = codegen::EquationPath;

    static std::vector<Access<VariableType, AccessProperty>> getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> accesses;

      for (const auto& access : (*equation)->getAccesses()) {
        bool isAllowed = true;

        auto allowedAccessFn = [&](mlir::Value value) {
          mlir::Operation* definingOp = value.getDefiningOp();

          if (!definingOp) {
            return true;
          }

          if (mlir::isa<mlir::modelica::SelectOp>(definingOp)) {
            isAllowed = false;
            return false;
          }

          return true;
        };

        const auto& path = access.getPath();
        (*equation)->traversePath(path, allowedAccessFn);

        if (isAllowed) {
          accesses.emplace_back(access.getVariable(), access.getAccessFunction(), path);
        }
      }

      return accesses;
    }
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATION_H
