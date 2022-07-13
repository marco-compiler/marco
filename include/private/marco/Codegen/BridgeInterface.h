#ifndef MARCO_CODEGEN_BRIDGEINTERFACE_H
#define MARCO_CODEGEN_BRIDGEINTERFACE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Results.h"

namespace marco::codegen::lowering
{
  class BridgeInterface
  {
    public:
      virtual ~BridgeInterface();

      virtual std::vector<mlir::Operation*> lower(const ast::Class& cls) = 0;

      virtual Results lower(const ast::Expression& expression) = 0;

      virtual void lower(const ast::Algorithm& algorithm) = 0;

      virtual void lower(const ast::Statement& statement) = 0;

      virtual void lower(const ast::Equation& equation, bool initialEquation) = 0;

      virtual void lower(const ast::ForEquation& forEquation, bool initialEquation) = 0;
  };
}

#endif // MARCO_CODEGEN_BRIDGEINTERFACE_H
