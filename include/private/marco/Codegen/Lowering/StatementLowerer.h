#ifndef MARCO_CODEGEN_LOWERING_STATEMENTBRIDGE_H
#define MARCO_CODEGEN_LOWERING_STATEMENTBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class StatementLowerer : public Lowerer
  {
    public:
      StatementLowerer(LoweringContext* context, BridgeInterface* bridge);

      void operator()(const ast::AssignmentStatement& statement);
      void operator()(const ast::IfStatement& statement);
      void operator()(const ast::ForStatement& statement);
      void operator()(const ast::WhileStatement& statement);
      void operator()(const ast::WhenStatement& statement);
      void operator()(const ast::BreakStatement& statement);
      void operator()(const ast::ReturnStatement& statement);
  };
}

#endif // MARCO_CODEGEN_LOWERING_STATEMENTBRIDGE_H
