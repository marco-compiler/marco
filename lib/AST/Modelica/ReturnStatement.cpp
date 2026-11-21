#include "marco/AST/Modelica/ReturnStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ReturnStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}
} // namespace marco::ast::modelica
