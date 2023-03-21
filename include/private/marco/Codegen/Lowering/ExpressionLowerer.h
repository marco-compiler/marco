#ifndef MARCO_CODEGEN_LOWERING_EXPRESSIONBRIDGE_H
#define MARCO_CODEGEN_LOWERING_EXPRESSIONBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/Lowering/CallLowerer.h"
#include "marco/Codegen/Lowering/OperationLowerer.h"
#include "marco/Codegen/BridgeInterface.h"
#include <memory>

namespace marco::codegen::lowering
{
  class ExpressionLowerer : public Lowerer
  {
    public:
      ExpressionLowerer(LoweringContext* context, BridgeInterface* bridge);

      Results operator()(const ast::Array& array);
      Results operator()(const ast::Call& call);
      Results operator()(const ast::Constant& constant);
      Results operator()(const ast::Operation& operation);
      Results operator()(const ast::ReferenceAccess& reference);
      Results operator()(const ast::Tuple& tuple);

    private:
      std::unique_ptr<CallLowerer> callLowerer;
      std::unique_ptr<OperationLowerer> operationLowerer;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EXPRESSIONBRIDGE_H
