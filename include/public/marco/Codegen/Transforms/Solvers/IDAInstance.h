#ifndef MARCO_CODEGEN_TRANSFORMS_SOLVERS_IDAINSTANCE_H
#define MARCO_CODEGEN_TRANSFORMS_SOLVERS_IDAINSTANCE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"

namespace mlir::modelica
{
  class IDAInstance
  {
    public:
    IDAInstance(
        llvm::StringRef identifier,
        mlir::SymbolTableCollection& symbolTableCollection,
        const DerivativesMap* derivativesMap,
        bool reducedSystem,
        bool reducedDerivatives,
        bool jacobianOneSweep,
        bool debugInformation);

    void setStartTime(double time);

    void setEndTime(double time);

    bool hasVariable(VariableOp variable) const;

    void addStateVariable(VariableOp variable);

    void addDerivativeVariable(VariableOp variable);

    void addAlgebraicVariable(VariableOp variable);

    bool hasEquation(ScheduledEquationInstanceOp equation) const;

    void addEquation(ScheduledEquationInstanceOp equation);

    mlir::LogicalResult declareInstance(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp);

    mlir::LogicalResult initialize(
        mlir::OpBuilder& builder,
        mlir::Location loc);

    mlir::LogicalResult configure(
        mlir::IRRewriter& rewriter,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        ModelOp modelOp,
        llvm::ArrayRef<VariableOp> variableOps,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        llvm::ArrayRef<SCCGroupOp> sccGroups);

    mlir::LogicalResult performCalcIC(
        mlir::OpBuilder& builder,
        mlir::Location loc);

    mlir::LogicalResult performStep(
        mlir::OpBuilder& builder,
        mlir::Location loc);

    mlir::Value getCurrentTime(
        mlir::OpBuilder& builder,
        mlir::Location loc);

    mlir::LogicalResult deleteInstance(
        mlir::OpBuilder& builder,
        mlir::Location loc);

    private:
    bool hasAlgebraicVariable(VariableOp variable) const;

    bool hasStateVariable(VariableOp variable) const;

    bool hasDerivativeVariable(VariableOp variable) const;

    mlir::LogicalResult addVariablesToIDA(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        llvm::ArrayRef<VariableOp> variableOps,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap);

    mlir::sundials::VariableGetterOp createGetterFunction(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        GlobalVariableOp variable,
        llvm::StringRef functionName);

    mlir::sundials::VariableSetterOp createSetterFunction(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        GlobalVariableOp variable,
        llvm::StringRef functionName);

    mlir::LogicalResult addEquationsToIDA(
        mlir::IRRewriter& rewriter,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        ModelOp modelOp,
        llvm::ArrayRef<VariableOp> variableOps,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        llvm::ArrayRef<SCCGroupOp> sccGroups,
        llvm::DenseMap<
            mlir::AffineMap,
            mlir::sundials::AccessFunctionOp>& accessFunctionsMap);

    mlir::LogicalResult addVariableAccessesInfoToIDA(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        ModelOp modelOp,
        ScheduledEquationInstanceOp equationOp,
        mlir::Value idaEquation,
        llvm::DenseMap<
            mlir::AffineMap,
            mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
        size_t& accessFunctionsCounter);

    mlir::sundials::AccessFunctionOp getOrCreateAccessFunction(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        mlir::AffineMap access,
        llvm::StringRef functionNamePrefix,
        llvm::DenseMap<
            mlir::AffineMap,
            mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
        size_t& accessFunctionsCounter);

    mlir::sundials::AccessFunctionOp createAccessFunction(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp moduleOp,
        mlir::AffineMap access,
        llvm::StringRef functionName);

    mlir::LogicalResult createResidualFunction(
        mlir::OpBuilder& builder,
        mlir::ModuleOp moduleOp,
        ScheduledEquationInstanceOp equationOp,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        mlir::Value idaEquation,
        llvm::StringRef residualFunctionName);

    mlir::LogicalResult getIndependentVariablesForAD(
        llvm::DenseSet<VariableOp>& result,
        ModelOp modelOp,
        ScheduledEquationInstanceOp equationOp);

    mlir::LogicalResult createPartialDerTemplateFunction(
        mlir::IRRewriter& rewriter,
        mlir::ModuleOp moduleOp,
        llvm::ArrayRef<VariableOp> variableOps,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        ScheduledEquationInstanceOp equationOp,
        const llvm::DenseSet<VariableOp>& independentVariables,
        llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
        llvm::StringRef templateName);

    mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
        mlir::IRRewriter& rewriter,
        mlir::ModuleOp moduleOp,
        llvm::ArrayRef<VariableOp> variableOps,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        ScheduledEquationInstanceOp equationOp,
        const llvm::DenseSet<VariableOp>& independentVariables,
        llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
        llvm::StringRef templateName);

    mlir::LogicalResult createJacobianFunction(
        mlir::OpBuilder& builder,
        mlir::ModuleOp moduleOp,
        ModelOp modelOp,
        ScheduledEquationInstanceOp equationOp,
        const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
        llvm::StringRef jacobianFunctionName,
        const llvm::DenseSet<VariableOp>& independentVariables,
        const llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
        VariableOp independentVariable,
        llvm::StringRef partialDerTemplateName);

    std::string getIDAFunctionName(llvm::StringRef name) const;

    std::optional<mlir::SymbolRefAttr>
    getDerivative(mlir::SymbolRefAttr variable) const;

    std::optional<mlir::SymbolRefAttr>
    getDerivedVariable(mlir::SymbolRefAttr derivative) const;

    mlir::LogicalResult getWritesMap(
        ModelOp modelOp,
        llvm::ArrayRef<SCCGroupOp> sccGroups,
        std::multimap<VariableOp, std::pair<
            IndexSet, ScheduledEquationInstanceOp>>& writesMap) const;

    private:
    /// Instance identifier.
    /// It is used to create unique symbols.
    std::string identifier;

    mlir::SymbolTableCollection* symbolTableCollection;

    const DerivativesMap* derivativesMap;

    bool reducedSystem;
    bool reducedDerivatives;
    bool jacobianOneSweep;
    bool debugInformation;

    std::optional<double> startTime;
    std::optional<double> endTime;

    /// The algebraic variables of the model that are managed by IDA.
    /// An algebraic variable is a variable that is not a parameter, state or
    /// derivative.
    llvm::SmallVector<VariableOp> algebraicVariables;

    /// The state variables of the model that are managed by IDA.
    /// A state variable is a variable for which there exists a derivative
    /// variable.
    llvm::SmallVector<VariableOp> stateVariables;

    /// The derivative variables of the model that are managed by IDA.
    /// A derivative variable is a variable that is the derivative of another
    /// variable.
    llvm::SmallVector<VariableOp> derivativeVariables;

    /// The SSA values of the IDA variables representing the algebraic ones.
    llvm::SmallVector<mlir::Value> idaAlgebraicVariables;

    /// The SSA values of the IDA variables representing the state ones.
    llvm::SmallVector<mlir::Value> idaStateVariables;

    /// Map used for a faster lookup of the algebraic variable position.
    llvm::DenseMap<VariableOp, size_t> algebraicVariablesLookup;

    /// Map used for a faster lookup of the state variable position.
    llvm::DenseMap<VariableOp, size_t> stateVariablesLookup;

    /// Map used for a faster lookup of the derivative variable position.
    llvm::DenseMap<VariableOp, size_t> derivativeVariablesLookup;

    /// The equations managed by IDA.
    llvm::DenseSet<ScheduledEquationInstanceOp> equations;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_SOLVERS_IDAINSTANCE_H
