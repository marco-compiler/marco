#ifndef MARCO_CODEGEN_LOWERING_EXTERNALFUNCTIONCALLLOWERER_H
#define MARCO_CODEGEN_LOWERING_EXTERNALFUNCTIONCALLLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
 class ExpressionLowerer;

  class ExternalFunctionCallLowerer : public Lowerer {
  public: 
    explicit ExternalFunctionCallLowerer(BridgeInterface *bridge);

    virtual bool lower(const ast::ExternalFunctionCall &call, mlir::bmodelica::FunctionOp *functionOp) override;
  
  protected:
    using Lowerer::lower;

  private:
    std::optional<mlir::Operation *>
  resolveCallee(const ast::ComponentReference &callee);

  std::optional<mlir::Value> lowerArg(const ast::Expression &expression);

  void getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      mlir::bmodelica::FunctionOp functionOp);

  [[nodiscard]] bool lowerCustomFunctionArgs(
      const ast::ExternalFunctionCall &call,
      llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues);


  /// Get the argument expected ranks of a user-defined function.
  void getFunctionExpectedArgRanks(mlir::Operation *op,
                                   llvm::SmallVectorImpl<int64_t> &ranks);

  /// Get the result types of a user-defined function.
  void getFunctionResultTypes(mlir::Operation *op,
                              llvm::SmallVectorImpl<mlir::Type> &types);

  /// Get the result type in case of a possibly element-wise call.
  /// The arguments are needed because some functions (such as min / size)
  /// may vary their behaviour according to arguments count.
  

  /// Helper function to emit an error if a function is provided the wrong
  /// number of arguments. The error will state that the function received
  /// actualNum arguments, but the expected number was exactly expectedNum.
  void emitErrorNumArguments(llvm::StringRef function,
                             const marco::SourceRange &location,
                             unsigned int actualNum, unsigned int expectedNum);

  };
}

#endif // MARCO_CODEGEN_LOWERING_EXTERNALFUNCTIONCALLLOWERER_H
