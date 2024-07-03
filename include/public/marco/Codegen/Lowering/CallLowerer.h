#ifndef MARCO_CODEGEN_LOWERING_CALLLOWERER_H
#define MARCO_CODEGEN_LOWERING_CALLLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include <functional>

namespace marco::codegen::lowering
{
  class ExpressionLowerer;

  class CallLowerer : public Lowerer
  {
    public:
      explicit CallLowerer(BridgeInterface* bridge);

      virtual std::optional<Results> lower(const ast::Call& call) override;

    protected:
      using Lowerer::lower;

    private:
      std::optional<mlir::Operation*> resolveCallee(
        const ast::ComponentReference& callee);

      std::optional<mlir::Value> lowerArg(const ast::Expression& expression);

      void getCustomFunctionInputVariables(
          llvm::SmallVectorImpl<mlir::bmodelica::VariableOp>& inputVariables,
          mlir::bmodelica::FunctionOp functionOp);

      void getCustomFunctionInputVariables(
          llvm::SmallVectorImpl<mlir::bmodelica::VariableOp>& inputVariables,
          mlir::bmodelica::DerFunctionOp derFunctionOp);

      [[nodiscard]] bool lowerCustomFunctionArgs(
          const ast::Call& call,
          llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
          llvm::SmallVectorImpl<std::string>& argNames,
          llvm::SmallVectorImpl<mlir::Value>& argValues);

      void getRecordConstructorInputVariables(
          llvm::SmallVectorImpl<mlir::bmodelica::VariableOp>& inputVariables,
          mlir::bmodelica::RecordOp recordOp);

      [[nodiscard]] bool lowerRecordConstructorArgs(
          const ast::Call& call,
          llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
          llvm::SmallVectorImpl<std::string>& argNames,
          llvm::SmallVectorImpl<mlir::Value>& argValues);

      [[nodiscard]] bool lowerBuiltInFunctionArgs(
          const ast::Call& call,
          llvm::SmallVectorImpl<mlir::Value>& args);

      std::optional<mlir::Value> lowerBuiltInFunctionArg(
          const ast::FunctionArgument& arg);

      /// Get the argument expected ranks of a user-defined function.
      void getFunctionExpectedArgRanks(
          mlir::Operation* op,
          llvm::SmallVectorImpl<int64_t>& ranks);

      /// Get the result types of a user-defined function.
      void getFunctionResultTypes(
          mlir::Operation* op,
          llvm::SmallVectorImpl<mlir::Type>& types);

      /// Get the result type in case of a possibly element-wise call.
      /// The arguments are needed because some functions (such as min / size)
      /// may vary their behaviour according to arguments count.
      bool getVectorizedResultTypes(
          llvm::ArrayRef<mlir::Value> args,
          llvm::ArrayRef<int64_t> expectedArgRanks,
          llvm::ArrayRef<mlir::Type> scalarizedResultTypes,
          llvm::SmallVectorImpl<mlir::Type>& inferredResultTypes) const;

      /// Check if a built-in function with a given name exists.
      bool isBuiltInFunction(const ast::ComponentReference& name) const;

      /// Emit an error due to a wrong number of arguments being provided to a function.
      void emitErrorNumArguments(const std::string &function, const marco::SourceRange& location, 
                                 unsigned int actualNum, unsigned int expectedNum);
      void emitErrorNumArgumentsRange(const std::string &function, const marco::SourceRange& location,
                                      unsigned int actualNum, unsigned int minExpectedNum, 
                                      unsigned int maxExpectedNum = 0);

      std::optional<Results> dispatchBuiltInFunctionCall(const ast::Call& call);

      std::optional<Results> abs(const ast::Call& call);
      std::optional<Results> acos(const ast::Call& call);
      std::optional<Results> asin(const ast::Call& call);
      std::optional<Results> atan(const ast::Call& call);
      std::optional<Results> atan2(const ast::Call& call);
      std::optional<Results> ceil(const ast::Call& call);
      std::optional<Results> cos(const ast::Call& call);
      std::optional<Results> cosh(const ast::Call& call);
      std::optional<Results> der(const ast::Call& call);
      std::optional<Results> diagonal(const ast::Call& call);
      std::optional<Results> div(const ast::Call& call);
      std::optional<Results> exp(const ast::Call& call);
      std::optional<Results> fill(const ast::Call& call);
      std::optional<Results> floor(const ast::Call& call);
      std::optional<Results> identity(const ast::Call& call);
      std::optional<Results> integer(const ast::Call& call);
      std::optional<Results> linspace(const ast::Call& call);
      std::optional<Results> log(const ast::Call& call);
      std::optional<Results> log10(const ast::Call& call);
      std::optional<Results> max(const ast::Call& call);
      std::optional<Results> maxArray(const ast::Call& call);
      std::optional<Results> maxReduction(const ast::Call& call);
      std::optional<Results> maxScalars(const ast::Call& call);
      std::optional<Results> min(const ast::Call& call);
      std::optional<Results> minArray(const ast::Call& call);
      std::optional<Results> minReduction(const ast::Call& call);
      std::optional<Results> minScalars(const ast::Call& call);
      std::optional<Results> mod(const ast::Call& call);
      std::optional<Results> ndims(const ast::Call& call);
      std::optional<Results> ones(const ast::Call& call);
      std::optional<Results> product(const ast::Call& call);
      std::optional<Results> productArray(const ast::Call& call);
      std::optional<Results> productReduction(const ast::Call& call);
      std::optional<Results> rem(const ast::Call& call);
      std::optional<Results> sign(const ast::Call& call);
      std::optional<Results> sin(const ast::Call& call);
      std::optional<Results> sinh(const ast::Call& call);
      std::optional<Results> size(const ast::Call& call);
      std::optional<Results> sqrt(const ast::Call& call);
      std::optional<Results> sum(const ast::Call& call);
      std::optional<Results> sumArray(const ast::Call& call);
      std::optional<Results> sumReduction(const ast::Call& call);
      std::optional<Results> symmetric(const ast::Call& call);
      std::optional<Results> tan(const ast::Call& call);
      std::optional<Results> tanh(const ast::Call& call);
      std::optional<Results> transpose(const ast::Call& call);
      std::optional<Results> zeros(const ast::Call& call);

      std::optional<Results> reduction(const ast::Call& call, llvm::StringRef action);
  };
}

#endif // MARCO_CODEGEN_LOWERING_CALLLOWERER_H
