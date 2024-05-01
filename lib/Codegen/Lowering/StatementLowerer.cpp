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

  void StatementLowerer::lower(const ast::Statement& statement)
  {
    if (auto casted = statement.dyn_cast<ast::AssignmentStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::BreakStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::ForStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::IfStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::ReturnStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::WhileStatement>()) {
      return lower(*casted);
    }

    if (auto casted = statement.dyn_cast<ast::WhenStatement>()) {
      return lower(*casted);
    }

    llvm_unreachable("Unknown statement type");
  }
}
