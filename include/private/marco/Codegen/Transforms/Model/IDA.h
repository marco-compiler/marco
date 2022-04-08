#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H

#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/Model/ExternalSolver.h"
#include <set>

namespace marco::codegen
{
  class IDASolver : public ExternalSolver
  {
    private:
      static constexpr size_t idaInstancePosition = 0;
      static constexpr size_t variablesOffset = 1;

    public:
      IDASolver(mlir::TypeConverter* typeConverter);

      bool isEnabled() const override;

      void setEnabled(bool status) override;

      bool containsEquation(ScheduledEquation* equation) const override;

      mlir::Type getRuntimeDataType(mlir::MLIRContext* context) override;

      bool hasVariable(mlir::Value variable) const;

      void addVariable(mlir::Value variable);

      bool hasEquation(ScheduledEquation* equation) const;

      void addEquation(ScheduledEquation* equation);

      mlir::LogicalResult processInitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::FuncOp initFunction,
          mlir::ValueRange variables,
          const Model<ScheduledEquationsBlock>& model,
          const mlir::BlockAndValueMapping& derivatives) override;

      mlir::LogicalResult processDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::FuncOp deinitFunction) override;

      mlir::LogicalResult processUpdateStatesFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::FuncOp updateStatesFunction,
          mlir::ValueRange variables,
          const mlir::BlockAndValueMapping& derivatives,
          double requestedTimeStep) override;

    private:
      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value);

      mlir::Value loadRuntimeData(
        mlir::OpBuilder& builder,
        mlir::Value runtimeDataPtr);

      void storeRuntimeData(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::Value value);

      mlir::Value getValueFromRuntimeData(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          mlir::Type type,
          unsigned int position);

      mlir::Value getIDAInstance(
          mlir::OpBuilder& builder, mlir::Value runtimeData);

      mlir::Value getIDAVariable(
          mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position);

      mlir::Value setIDAInstance(
          mlir::OpBuilder& builder, mlir::Value runtimeData, mlir::Value instance);

      mlir::Value setIDAVariable(
          mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position, mlir::Value variable);

      mlir::LogicalResult addVariablesToIDA(
        mlir::OpBuilder& builder,
        mlir::Value runtimeDataPtr,
        mlir::ValueRange variables,
        const mlir::BlockAndValueMapping& derivatives);

      mlir::LogicalResult addEquationsToIDA(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          const Model<ScheduledEquationsBlock>& model,
          const mlir::BlockAndValueMapping& derivatives);

      mlir::LogicalResult addVariableAccessesInfoToIDA(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          const Equation& equation,
          mlir::Value idaEquation);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::ValueRange equationVariables,
          mlir::Value idaEquation,
          llvm::StringRef residualFunctionName);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::ValueRange equationVariables,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::ValueRange equationVariables,
          const mlir::BlockAndValueMapping& derivatives,
          llvm::StringRef jacobianFunctionName,
          mlir::Value independentVariable,
          llvm::StringRef partialDerTemplateName);

    private:
      bool enabled;

      /// The variables of the model that are managed by IDA.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> managedVariables;

      std::set<ScheduledEquation*> equations;

      /// Map from a ModelOp variable index to its IDA variable index
      std::map<unsigned int, unsigned int> mappedVariables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H
