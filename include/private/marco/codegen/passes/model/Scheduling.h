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
          ::marco::modeling::scheduling::Direction schedulingDirection);

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

      const Variables& getVariables() const override;

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
          const ::marco::modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath,
          const Access& sourceAccess) const override;

      mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          ::marco::modeling::scheduling::Direction iterationDirection) const override;

      /// }
      /// @name Modified methods
      /// {

      size_t getNumOfIterationVars() const override;
      long getRangeBegin(size_t inductionVarIndex) const override;
      long getRangeEnd(size_t inductionVarIndex) const override;

      /// }

      /// Get the direction to be used to update the iteration variables.
      ::marco::modeling::scheduling::Direction getSchedulingDirection() const;

      /// Set the scheduled indexes
      void setScheduledIndexes(size_t inductionVarIndex, long begin, long end);

    private:
      std::unique_ptr<MatchedEquation> equation;
      std::vector<std::pair<long, long>> scheduledIndexes;
      ::marco::modeling::scheduling::Direction schedulingDirection;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_SCHEDULING_H
