#include "marco/Codegen/Lowering/StatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  StatementLowerer::StatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  __attribute__((warn_unused_result)) bool StatementLowerer::lower(const ast::Statement& statement)
  {
    if (auto casted = statement.dyn_cast<ast::AssignmentStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::BreakStatement>()) {
      lower(*casted);
      return true;
    }

    if (auto casted = statement.dyn_cast<ast::ForStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::IfStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::ReturnStatement>()) {
      lower(*casted);
      return true;
    }

    if (auto casted = statement.dyn_cast<ast::WhileStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::WhenStatement>()) {
      lower(*casted);
      return true;
    }

    llvm_unreachable("Unknown statement type");
  }
}
