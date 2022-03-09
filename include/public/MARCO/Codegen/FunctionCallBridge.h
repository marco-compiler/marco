#ifndef MARCO_CODEGEN_FUNCTIONCALLBRIDGE_H
#define MARCO_CODEGEN_FUNCTIONCALLBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/NewBridge.h"
#include <functional>

namespace marco::codegen
{
  class FunctionCallBridge
  {
    public:
      using Lowerer = std::function<NewLoweringBridge::Results(FunctionCallBridge&, const ast::Call&)>;
      using Results = NewLoweringBridge::Results;

      FunctionCallBridge(NewLoweringBridge* bridge);

      Results userDefinedFunction(const ast::Call& call);

      Results abs(const ast::Call& call);
      Results acos(const ast::Call& call);
      Results asin(const ast::Call& call);
      Results atan(const ast::Call& call);
      Results atan2(const ast::Call& call);
      Results cos(const ast::Call& call);
      Results cosh(const ast::Call& call);
      Results der(const ast::Call& call);
      Results diagonal(const ast::Call& call);
      Results exp(const ast::Call& call);
      Results identity(const ast::Call& call);
      Results linspace(const ast::Call& call);
      Results log(const ast::Call& call);
      Results log10(const ast::Call& call);
      Results max(const ast::Call& call);
      Results min(const ast::Call& call);
      Results ndims(const ast::Call& call);
      Results ones(const ast::Call& call);
      Results product(const ast::Call& call);
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

    private:
      NewLoweringBridge* bridge;
  };
}

#endif // MARCO_CODEGEN_FUNCTIONCALLBRIDGE_H
