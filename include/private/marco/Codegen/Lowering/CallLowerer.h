#ifndef MARCO_CODEGEN_LOWERING_CALLLOWERER_H
#define MARCO_CODEGEN_LOWERING_CALLLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/BridgeInterface.h"
#include <functional>

namespace marco::codegen::lowering
{
  class ExpressionLowerer;

  class CallLowerer : public Lowerer
  {
    public:
      CallLowerer(BridgeInterface* bridge);

      virtual Results lower(const ast::Call& call) override;

    protected:
      using Lowerer::lower;

    private:
      std::optional<mlir::Operation*> resolveCallee(
        const ast::ComponentReference& callee);

      mlir::Value lowerArg(const ast::Expression& expression);

      void lowerArgs(
          const ast::Call& call,
          llvm::SmallVectorImpl<mlir::Value>& args);

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

      Results dispatchBuiltInFunctionCall(const ast::Call& call);

      Results abs(const ast::Call& call);
      Results acos(const ast::Call& call);
      Results asin(const ast::Call& call);
      Results atan(const ast::Call& call);
      Results atan2(const ast::Call& call);
      Results ceil(const ast::Call& call);
      Results cos(const ast::Call& call);
      Results cosh(const ast::Call& call);
      Results der(const ast::Call& call);
      Results diagonal(const ast::Call& call);
      Results div(const ast::Call& call);
      Results exp(const ast::Call& call);
      Results fill(const ast::Call& call);
      Results floor(const ast::Call& call);
      Results identity(const ast::Call& call);
      Results integer(const ast::Call& call);
      Results linspace(const ast::Call& call);
      Results log(const ast::Call& call);
      Results log10(const ast::Call& call);
      Results max(const ast::Call& call);
      Results min(const ast::Call& call);
      Results mod(const ast::Call& call);
      Results ndims(const ast::Call& call);
      Results ones(const ast::Call& call);
      Results product(const ast::Call& call);
      Results rem(const ast::Call& call);
      Results sign(const ast::Call& call);
      Results sin(const ast::Call& call);
      Results sinh(const ast::Call& call);
      Results size(const ast::Call& call);
      Results sqrt(const ast::Call& call);
      Results sum(const ast::Call& call);
      Results symmetric(const ast::Call& call);
      Results tan(const ast::Call& call);
      Results tanh(const ast::Call& call);
      Results transpose(const ast::Call& call);
      Results zeros(const ast::Call& call);
  };
}

#endif // MARCO_CODEGEN_LOWERING_CALLLOWERER_H
