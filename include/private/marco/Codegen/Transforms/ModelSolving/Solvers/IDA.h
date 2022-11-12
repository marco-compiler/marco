#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_IDA_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_IDA_H

#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Solvers/ModelSolver.h"

namespace marco::codegen
{
  class IDAInstance
  {
    private:
      static constexpr size_t idaInstancePosition = 0;
      static constexpr size_t variablesOffset = 1;

    public:
      IDAInstance(
          mlir::TypeConverter* typeConverter,
          const DerivativesMap& derivativesMap);

      void setStartTime(double time);

      void setEndTime(double time);

      bool hasVariable(mlir::Value variable) const;

      void addParametricVariable(mlir::Value variable);

      void addStateVariable(mlir::Value variable);

      void addDerivativeVariable(mlir::Value variable);

      void addAlgebraicVariable(mlir::Value variable);

      bool hasEquation(ScheduledEquation* equation) const;

      void addEquation(ScheduledEquation* equation);

      mlir::Type getSolverDataType(mlir::MLIRContext* context) const;

      mlir::LogicalResult createInstance(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr);

      mlir::LogicalResult configure(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr,
          const Model<ScheduledEquationsBlock>& model,
          mlir::ValueRange variables);

      mlir::LogicalResult performCalcIC(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr);

      mlir::LogicalResult performStep(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr);

      mlir::Value getCurrentTime(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr,
          mlir::Type timeType);

      mlir::LogicalResult deleteInstance(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr);

    private:
      bool hasParametricVariable(mlir::Value variable) const;

      bool hasAlgebraicVariable(mlir::Value variable) const;

      bool hasStateVariable(mlir::Value variable) const;

      bool hasDerivativeVariable(mlir::Value variable) const;

      mlir::Value materializeTargetConversion(
        mlir::OpBuilder& builder,
        mlir::Value value);

      mlir::Value loadSolverData(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr);

      mlir::Value getValueFromSolverData(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          mlir::Type type,
          unsigned int position);

      mlir::Value getIDAInstance(
          mlir::OpBuilder& builder,
          mlir::Value solverData);

      mlir::Value getIDAVariable(
          mlir::OpBuilder& builder,
          mlir::Value solverData,
          unsigned int position);

      void storeSolverData(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr,
          mlir::Value solverData);

      mlir::Value setIDAInstance(
          mlir::OpBuilder& builder,
          mlir::Value solverData,
          mlir::Value instance);

      mlir::Value setIDAVariable(
          mlir::OpBuilder& builder,
          mlir::Value solverData,
          unsigned int position,
          mlir::Value variable);

      mlir::LogicalResult addVariablesToIDA(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value runtimeDataPtr,
          mlir::ValueRange variables);

      mlir::ida::VariableGetterOp createGetterFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type variableType,
          llvm::StringRef functionName);

      mlir::ida::VariableSetterOp createSetterFunction(
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
          mlir::Value idaEquation,
          llvm::StringRef residualFunctionName);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          llvm::StringRef templateName);

      mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::OpBuilder& builder,
          const marco::codegen::Equation& equation,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          llvm::StringRef jacobianFunctionName,
          mlir::Value independentVariable,
          llvm::StringRef partialDerTemplateName);

      std::vector<mlir::Value> getIDAFunctionArgs() const;

    private:
      mlir::TypeConverter* typeConverter;
      const DerivativesMap* derivativesMap;

      llvm::Optional<double> startTime;
      llvm::Optional<double> endTime;

      /// The parametric variables of the model that are managed by IDA.
      /// A parametric variable is a variable that is immutable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> parametricVariables;

      /// The algebraic variables of the model that are managed by IDA.
      /// An algebraic variable is a variable that is not a parameter, state or
      /// derivative.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> algebraicVariables;

      /// The state variables of the model that are managed by IDA.
      /// A state variable is a variable for which there exists a derivative
      /// variable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> stateVariables;

      /// The derivative variables of the model that are managed by IDA.
      /// A derivative variable is a variable that is the derivative of another
      /// variable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> derivativeVariables;

      /// The SSA values of the IDA variables representing the algebraic ones.
      std::vector<mlir::Value> idaAlgebraicVariables;

      /// The SSA values of the IDA variables representing the state ones.
      std::vector<mlir::Value> idaStateVariables;

      /// Map used for a faster lookup of the parametric variable position
      /// given its number.
      llvm::DenseMap<unsigned int, size_t> parametricVariablesLookup;

      /// Map used for a faster lookup of the algebraic variable position given
      /// its number.
      llvm::DenseMap<unsigned int, size_t> algebraicVariablesLookup;

      /// Map used for a faster lookup of the state variable position given
      /// its number.
      llvm::DenseMap<unsigned int, size_t> stateVariablesLookup;

      /// Map used for a faster lookup of the derivative variable position
      /// given its number.
      llvm::DenseMap<unsigned int, size_t> derivativeVariablesLookup;

      /// The equations managed by IDA.
      std::set<ScheduledEquation*> equations;
  };

  class IDASolver : public ModelSolver
  {
    public:
      static constexpr llvm::StringLiteral initICSolversFunctionName = "initICSolvers";
      static constexpr llvm::StringLiteral deinitICSolversFunctionName = "deinitICSolvers";
      static constexpr llvm::StringLiteral solveICModelFunctionName = "solveICModel";
      static constexpr llvm::StringLiteral initMainSolversFunctionName = "initMainSolvers";
      static constexpr llvm::StringLiteral deinitMainSolversFunctionName = "deinitMainSolvers";
      static constexpr llvm::StringLiteral calcICFunctionName = "calcIC";
      static constexpr llvm::StringLiteral updateIDAVariablesFunctionName = "updateIDAVariables";
      static constexpr llvm::StringLiteral updateNonIDAVariablesFunctionName = "updateNonIDAVariables";
      static constexpr llvm::StringLiteral incrementTimeFunctionName = "incrementTime";

      struct ConversionInfo
      {
        std::set<std::unique_ptr<Equation>> explicitEquations;
        std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
        std::set<ScheduledEquation*> implicitEquations;
        std::set<ScheduledEquation*> cyclicEquations;
      };

      IDASolver(mlir::LLVMTypeConverter& typeConverter,
                VariableFilter& variablesFilter);

      mlir::LogicalResult solveICModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) override;

      mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) override;

    private:
      /// Create the function that instantiates the external solvers to be used
      /// during the IC computation.
      mlir::LogicalResult createInitICSolversFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers used during
      /// the IC computation.
      mlir::LogicalResult createDeinitICSolversFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers to be used
      /// during the simulation.
      mlir::LogicalResult createInitMainSolversFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers used during
      /// the simulation.
      mlir::LogicalResult createDeinitMainSolversFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers.
      mlir::LogicalResult createInitSolversFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers.
      mlir::LogicalResult createDeinitSolversFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the
      /// "initial conditions model".
      mlir::LogicalResult createSolveICModelFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the "main
      /// model".
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateIDAVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// not belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateNonIDAVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo,
          IDAInstance* idaInstance) const;

      /// Create the function to be used to increment the time.
      mlir::LogicalResult createIncrementTimeFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      mlir::func::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::func::FuncOp templateFunction,
          mlir::TypeRange varsTypes) const;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_IDA_H
