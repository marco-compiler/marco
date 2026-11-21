#include "marco/AST/Modelica/InheritanceModification.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void InheritanceModification::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(connectLhs, callback);
  walk(connectRhs, callback);
}

void InheritanceModification::addJSONProperties(
    llvm::json::Object &obj) const {
  if (isConnect()) {
    obj["connect_lhs"] = modelica::toJSON(connectLhs.get());
    obj["connect_rhs"] = modelica::toJSON(connectRhs.get());
  } else {
    obj["component"] = component;
  }
}

llvm::StringRef InheritanceModification::getComponent() const {
  return component;
}

void InheritanceModification::setComponent(std::string name) {
  component = std::move(name);
}

ComponentReference *InheritanceModification::getConnectLhs() const {
  return connectLhs.get();
}

ComponentReference *InheritanceModification::getConnectRhs() const {
  return connectRhs.get();
}

void InheritanceModification::setConnect(
    std::unique_ptr<ComponentReference> lhs,
    std::unique_ptr<ComponentReference> rhs) {
  connectLhs = std::move(lhs);
  connectRhs = std::move(rhs);
}

bool InheritanceModification::isConnect() const {
  return static_cast<bool>(connectLhs);
}
} // namespace marco::ast::modelica
