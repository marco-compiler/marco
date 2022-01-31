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

      modelica::EquationOp getOperation() const override;

      const Variables& getVariables() const override;

      void setVariables(Variables variables) override;

      std::vector<Access> getAccesses() const override;

      ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const override;

      std::vector<Access> getReads() const;

      Access getWrite() const;

      std::unique_ptr<Equation> explicitate(mlir::OpBuilder& builder);

      mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars) const override;

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
