#ifndef MARCO_CODEGEN_LOWERING_CALLBRIDGE_H
#define MARCO_CODEGEN_LOWERING_CALLBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/BridgeInterface.h"
#include <functional>

namespace marco::codegen::lowering
{
  class CallLowerer : public Lowerer
  {
    public:
      using LoweringFunction = std::function<Results(CallLowerer&, const ast::Call&)>;

      CallLowerer(LoweringContext* context, BridgeInterface* bridge);

      Results userDefinedFunction(const ast::Call& call);

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

#endif // MARCO_CODEGEN_LOWERING_CALLBRIDGE_H
