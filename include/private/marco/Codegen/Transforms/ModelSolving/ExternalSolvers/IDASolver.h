#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_IDASOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_IDASOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/ExternalSolver.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDAOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <set>

namespace marco::codegen
{
  enum class IDAVariableType
  {
    /// A variable that is neither a state nor a derivative.
    ALGEBRAIC,

    /// A variable for which it exists a derivative variable.
    STATE,

    /// A variable that is the derivative of another one.
    DERIVATIVE
  };

  struct IDAVariable
  {
    IDAVariable(unsigned int argNumber, IDAVariableType type);

    unsigned int argNumber;
    IDAVariableType type;
  };

  class IDASolver : public ExternalSolver
  {
    private:
      static constexpr size_t idaInstancePosition = 0;
      static constexpr size_t variablesOffset = 1;

    public:
      IDASolver(
        mlir::TypeConverter* typeConverter,
        const DerivativesMap& derivativesMap,
        IDAOptions options,
        double startTime, double endTime, double timeStep);

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
          const Model<ScheduledEquationsBlock>& model) override;

      mlir::LogicalResult processDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::FuncOp deinitFunction) override;

      mlir::LogicalResult processUpdateStatesFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::FuncOp updateStatesFunction,
          mlir::ValueRange variables) override;

      bool hasTimeOwnership() const override;

      mlir::Value getCurrentTime(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr) override;

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
        mlir::ModuleOp module,
        mlir::Value runtimeDataPtr,
        mlir::ValueRange variables);

      mlir::LogicalResult createGetterFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type variableType,
          llvm::StringRef functionName);

      mlir::LogicalResult createSetterFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type variableType,
          llvm::StringRef functionName);

      mlir::LogicalResult addEquationsToIDA(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          const Model<ScheduledEquationsBlock>& model);

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

      mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::OpBuilder& builder,
          const marco::codegen::Equation& equation,
          mlir::ValueRange originalVariables,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::ValueRange equationVariables,
          llvm::StringRef jacobianFunctionName,
          mlir::Value independentVariable,
          llvm::StringRef partialDerTemplateName);

      std::vector<mlir::Value> filterByManagedVariables(mlir::ValueRange variables) const;

    private:
      const DerivativesMap* derivativesMap;

      bool enabled;
      IDAOptions options;

      const double startTime;
      const double endTime;
      const double timeStep;

      /// The variables of the model that are managed by IDA.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<IDAVariable> managedVariables;

      /// The equations managed by IDA.
      std::set<ScheduledEquation*> equations;

      /// Map from the argument numbers of the elements of 'managedVariables'
      /// to the index of the IDA variables living within the IDA runtime
      /// data structure.
      std::map<unsigned int, size_t> mappedVariables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_IDASOLVER_H
