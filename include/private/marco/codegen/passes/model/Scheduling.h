#ifndef MARCO_CODEGEN_PASSES_MODEL_SCHEDULING_H
#define MARCO_CODEGEN_PASSES_MODEL_SCHEDULING_H

#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Matching.h"
#include "marco/modeling/Scheduling.h"
#include <memory>

namespace marco::codegen
{
  class ScheduledEquation : public Equation
  {
    public:
      ScheduledEquation(
          std::unique_ptr<MatchedEquation> equation,
          modeling::MultidimensionalRange scheduledIndexes,
          modeling::scheduling::Direction schedulingDirection);

      ScheduledEquation(const ScheduledEquation& other);

      ~ScheduledEquation();

      ScheduledEquation& operator=(const ScheduledEquation& other);
      ScheduledEquation& operator=(ScheduledEquation&& other);

      friend void swap(ScheduledEquation& first, ScheduledEquation& second);

      std::unique_ptr<Equation> clone() const override;

      /// @name Forwarded methods
      /// {

      modelica::EquationOp cloneIR() const override;

      void eraseIR() override;

      void dumpIR() const override;

      modelica::EquationOp getOperation() const override;

      Variables getVariables() const override;

      void setVariables(Variables variables) override;

      std::vector<Access> getAccesses() const override;

      ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const override;

      mlir::Value getValueAtPath(const EquationPath& path) const override;

      std::vector<Access> getReads() const;

      Access getWrite() const;

      mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder, const EquationPath& path) override;

      std::unique_ptr<Equation> cloneAndExplicitate(
          mlir::OpBuilder& builder, const EquationPath& path) const override;

      std::unique_ptr<Equation> cloneAndExplicitate(mlir::OpBuilder& builder) const;

      std::vector<mlir::Value> getInductionVariables() const override;

      mlir::LogicalResult replaceInto(
          mlir::OpBuilder& builder,
          Equation& destination,
          const modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath) const override;

      mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          modeling::scheduling::Direction iterationDirection) const override;

      /// }
      /// @name Modified methods
      /// {

      size_t getNumOfIterationVars() const override;

      modeling::MultidimensionalRange getIterationRanges() const override;

      /// }

      /// Get the direction to be used to update the iteration variables.
      ::marco::modeling::scheduling::Direction getSchedulingDirection() const;

    private:
      std::unique_ptr<MatchedEquation> equation;
      modeling::MultidimensionalRange scheduledIndexes;
      modeling::scheduling::Direction schedulingDirection;
  };

  // Specialize the container for equations in case of scheduled ones, in order to
  // force them to be ordered.
  template<>
  class Equations<ScheduledEquation>
  {
    private:
      using Impl = impl::Equations<ScheduledEquation, std::vector>;

    public:
      using iterator = typename Impl::iterator;
      using const_iterator = typename Impl::const_iterator;

      Equations() : impl(std::make_shared<Impl>())
      {
      }

      /// @name Forwarded methods
      /// {

      void push_back(std::unique_ptr<ScheduledEquation> equation)
      {
        impl->add(std::move(equation));
      }

      size_t size() const
      {
        return impl->size();
      }

      std::unique_ptr<ScheduledEquation>& operator[](size_t index)
      {
        return (*impl)[index];
      }

      const std::unique_ptr<ScheduledEquation>& operator[](size_t index) const
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

#endif // MARCO_CODEGEN_PASSES_MODEL_SCHEDULING_H
