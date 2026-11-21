#include "marco/AST/Modelica/Description.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Description::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(annotation, callback);
}

void Description::addToJSON(llvm::json::Object &obj) const {
  if (!text.empty()) {
    obj["description"] = text;
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}
} // namespace marco::ast::modelica
