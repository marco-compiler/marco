#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_KINSOLSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_KINSOLSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/ExternalSolver.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/KINSOLOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <set>

namespace marco::codegen
{
  struct KINSOLVariable
  {
    KINSOLVariable(unsigned int argNumber);

    unsigned int argNumber;
  };

  class KINSOLSolver : public ExternalSolver
  {
    private:
      static constexpr size_t kinsolInstancePosition = 0;
      static constexpr size_t variablesOffset = 1;

    public:
      KINSOLSolver(
        mlir::TypeConverter* typeConverter,
        KINSOLOptions options);

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
          mlir::func::FuncOp initFunction,
          mlir::ValueRange variables,
          const Model<ScheduledEquationsBlock>& model) override;

      mlir::LogicalResult processDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::func::FuncOp deinitFunction) override;

      mlir::LogicalResult processUpdateStatesFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::func::FuncOp updateStatesFunction,
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

      mlir::Value getKINSOLInstance(
          mlir::OpBuilder& builder, mlir::Value runtimeData);

      mlir::Value getKINSOLVariable(
          mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position);

      mlir::Value setKINSOLInstance(
          mlir::OpBuilder& builder, mlir::Value runtimeData, mlir::Value instance);

      mlir::Value setKINSOLVariable(
          mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position, mlir::Value variable);

      mlir::LogicalResult addVariablesToKINSOL(
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

      mlir::LogicalResult addEquationsToKINSOL(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          const Model<ScheduledEquationsBlock>& model);

      mlir::LogicalResult addVariableAccessesInfoToKINSOL(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          const Equation& equation,
          mlir::Value kinsolEquation);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::ValueRange equationVariables,
          mlir::Value kinsolEquation,
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
      bool enabled;
      KINSOLOptions options;

      /// The variables of the model that are managed by KINSOL.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<KINSOLVariable> managedVariables;

      /// The equations managed by KINSOL.
      std::set<ScheduledEquation*> equations;

      /// Map from the argument numbers of the elements of 'managedVariables'
      /// to the index of the KINSOL variables living within the KINSOL runtime
      /// data structure.
      std::map<unsigned int, size_t> mappedVariables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_KINSOLSOLVER_H
