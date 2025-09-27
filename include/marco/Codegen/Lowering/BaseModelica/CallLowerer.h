#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/Results.h"
#include <functional>

namespace marco::codegen::lowering::bmodelica {
class ExpressionLowerer;

class CallLowerer : public Lowerer {
public:
  explicit CallLowerer(BridgeInterface *bridge);

  virtual std::optional<Results>
  lower(const ast::bmodelica::Call &call) override;

protected:
  using Lowerer::lower;

private:
  std::optional<mlir::Operation *>
  resolveCallee(const ast::bmodelica::ComponentReference &callee);

  std::optional<mlir::Value>
  lowerArg(const ast::bmodelica::Expression &expression);

  void getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      mlir::bmodelica::FunctionOp functionOp);

  void getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      mlir::bmodelica::DerFunctionOp derFunctionOp);

  [[nodiscard]] bool lowerCustomFunctionArgs(
      const ast::bmodelica::Call &call,
      llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues);

  void getRecordConstructorInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      mlir::bmodelica::RecordOp recordOp);

  [[nodiscard]] bool lowerRecordConstructorArgs(
      const ast::bmodelica::Call &call,
      llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues);

  [[nodiscard]] bool
  lowerBuiltInFunctionArgs(const ast::bmodelica::Call &call,
                           llvm::SmallVectorImpl<mlir::Value> &args);

  std::optional<mlir::Value>
  lowerBuiltInFunctionArg(const ast::bmodelica::FunctionArgument &arg);

  /// Get the argument expected ranks of a user-defined function.
  void getFunctionExpectedArgRanks(mlir::Operation *op,
                                   llvm::SmallVectorImpl<int64_t> &ranks);

  /// Get the result types of a user-defined function.
  void getFunctionResultTypes(mlir::Operation *op,
                              llvm::SmallVectorImpl<mlir::Type> &types);

  /// Get the result type in case of a possibly element-wise call.
  /// The arguments are needed because some functions (such as min / size)
  /// may vary their behaviour according to arguments count.
  bool getVectorizedResultTypes(
      llvm::ArrayRef<mlir::Value> args,
      llvm::ArrayRef<int64_t> expectedArgRanks,
      llvm::ArrayRef<mlir::Type> scalarizedResultTypes,
      llvm::SmallVectorImpl<mlir::Type> &inferredResultTypes) const;

  /// Check if a built-in function with a given name exists.
  bool isBuiltInFunction(const ast::bmodelica::ComponentReference &name) const;

  /// Helper function to emit an error if a function is provided the wrong
  /// number of arguments. The error will state that the function received
  /// actualNum arguments, but the expected number was exactly expectedNum.
  void emitErrorNumArguments(llvm::StringRef function,
                             const marco::SourceRange &location,
                             unsigned int actualNum, unsigned int expectedNum);

  /// Helper function to emit an error if a function is provided the wrong
  /// number of arguments. The error will state that the function received
  /// actualNum arguments. If maxExpectedNum is 0, the function will state that
  /// the expected number of arguments was at least minExpectedNum, otherwise it
  /// will state that the expected number of arguments was in the range
  /// [minExpectedNum, maxExpectedNum].
  void emitErrorNumArgumentsRange(llvm::StringRef function,
                                  const marco::SourceRange &location,
                                  unsigned int actualNum,
                                  unsigned int minExpectedNum,
                                  unsigned int maxExpectedNum = 0);

  std::optional<Results>
  dispatchBuiltInFunctionCall(const ast::bmodelica::Call &call);

  std::optional<Results> abs(const ast::bmodelica::Call &call);
  std::optional<Results> acos(const ast::bmodelica::Call &call);
  std::optional<Results> asin(const ast::bmodelica::Call &call);
  std::optional<Results> atan(const ast::bmodelica::Call &call);
  std::optional<Results> atan2(const ast::bmodelica::Call &call);
  std::optional<Results> ceil(const ast::bmodelica::Call &call);
  std::optional<Results> cos(const ast::bmodelica::Call &call);
  std::optional<Results> cosh(const ast::bmodelica::Call &call);
  std::optional<Results> der(const ast::bmodelica::Call &call);
  std::optional<Results> diagonal(const ast::bmodelica::Call &call);
  std::optional<Results> div(const ast::bmodelica::Call &call);
  std::optional<Results> exp(const ast::bmodelica::Call &call);
  std::optional<Results> fill(const ast::bmodelica::Call &call);
  std::optional<Results> floor(const ast::bmodelica::Call &call);
  std::optional<Results> identity(const ast::bmodelica::Call &call);
  std::optional<Results> integer(const ast::bmodelica::Call &call);
  std::optional<Results> linspace(const ast::bmodelica::Call &call);
  std::optional<Results> log(const ast::bmodelica::Call &call);
  std::optional<Results> log10(const ast::bmodelica::Call &call);
  std::optional<Results> max(const ast::bmodelica::Call &call);
  std::optional<Results> maxArray(const ast::bmodelica::Call &call);
  std::optional<Results> maxReduction(const ast::bmodelica::Call &call);
  std::optional<Results> maxScalars(const ast::bmodelica::Call &call);
  std::optional<Results> min(const ast::bmodelica::Call &call);
  std::optional<Results> minArray(const ast::bmodelica::Call &call);
  std::optional<Results> minReduction(const ast::bmodelica::Call &call);
  std::optional<Results> minScalars(const ast::bmodelica::Call &call);
  std::optional<Results> mod(const ast::bmodelica::Call &call);
  std::optional<Results> ndims(const ast::bmodelica::Call &call);
  std::optional<Results> ones(const ast::bmodelica::Call &call);
  std::optional<Results> product(const ast::bmodelica::Call &call);
  std::optional<Results> productArray(const ast::bmodelica::Call &call);
  std::optional<Results> productReduction(const ast::bmodelica::Call &call);
  std::optional<Results> rem(const ast::bmodelica::Call &call);
  std::optional<Results> sign(const ast::bmodelica::Call &call);
  std::optional<Results> sin(const ast::bmodelica::Call &call);
  std::optional<Results> sinh(const ast::bmodelica::Call &call);
  std::optional<Results> size(const ast::bmodelica::Call &call);
  std::optional<Results> sqrt(const ast::bmodelica::Call &call);
  std::optional<Results> sum(const ast::bmodelica::Call &call);
  std::optional<Results> sumArray(const ast::bmodelica::Call &call);
  std::optional<Results> sumReduction(const ast::bmodelica::Call &call);
  std::optional<Results> symmetric(const ast::bmodelica::Call &call);
  std::optional<Results> tan(const ast::bmodelica::Call &call);
  std::optional<Results> tanh(const ast::bmodelica::Call &call);
  std::optional<Results> transpose(const ast::bmodelica::Call &call);
  std::optional<Results> zeros(const ast::bmodelica::Call &call);

  std::optional<Results> builtinAssert(const ast::bmodelica::Call &call);

  std::optional<Results> reduction(const ast::bmodelica::Call &call,
                                   llvm::StringRef action);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLLOWERER_H
