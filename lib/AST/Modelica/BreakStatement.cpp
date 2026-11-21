#include "marco/AST/Modelica/BreakStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void BreakStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}
} // namespace marco::ast::modelica
