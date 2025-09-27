#include "marco/Codegen/Lowering/BaseModelica/StatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
StatementLowerer::StatementLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool StatementLowerer::lower(const ast::bmodelica::Statement &statement) {
  if (auto casted = statement.dyn_cast<ast::bmodelica::AssignmentStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::BreakStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::CallStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::ForStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::IfStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::ReturnStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::WhileStatement>()) {
    return lower(*casted);
  }

  if (auto casted = statement.dyn_cast<ast::bmodelica::WhenStatement>()) {
    return lower(*casted);
  }

  llvm_unreachable("Unknown statement type");
}
} // namespace marco::codegen::lowering::bmodelica
