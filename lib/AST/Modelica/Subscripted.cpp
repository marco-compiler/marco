#include "marco/AST/Modelica/Subscripted.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Subscripted::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(base, callback);
  walk(subscripts, callback);
}

void Subscripted::addJSONProperties(llvm::json::Object &obj) const {
  obj["base"] = modelica::toJSON(base.get());
  obj["subscripts"] = modelica::toJSON(subscripts);
}

Expression *Subscripted::getBase() const { return base.get(); }

void Subscripted::setBase(std::unique_ptr<Expression> node) {
  base = std::move(node);
}

ChildrenRef<Expression> Subscripted::getSubscripts() const {
  return subscripts;
}

void Subscripted::addSubscript(std::unique_ptr<Expression> node) {
  subscripts.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
