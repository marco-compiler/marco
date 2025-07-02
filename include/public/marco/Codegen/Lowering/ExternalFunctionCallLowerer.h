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

    virtual std::optional<Results> lower(const ast::ExternalFunctionCall &call) override;
  
  protected:
    using Lowerer::lower;

  };
}

#endif // MARCO_CODEGEN_LOWERING_EXTERNALFUNCTIONCALLLOWERER_H
