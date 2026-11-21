#include "marco/AST/Modelica/ForIndex.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ForIndex::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(range, callback);
}

void ForIndex::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  obj["range"] = modelica::toJSON(range.get());
}

llvm::StringRef ForIndex::getName() const { return name; }

void ForIndex::setName(std::string newName) { name = std::move(newName); }

Expression *ForIndex::getRange() const { return range.get(); }

void ForIndex::setRange(std::unique_ptr<Expression> node) {
  range = std::move(node);
}
} // namespace marco::ast::modelica
