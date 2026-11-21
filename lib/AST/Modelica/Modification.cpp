#include "marco/AST/Modelica/Modification.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Modification::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(classModification, callback);
  walk(value, callback);
}

void Modification::addJSONProperties(llvm::json::Object &obj) const {
  if (classModification) {
    obj["class_modification"] = modelica::toJSON(classModification.get());
  }

  if (breakValue) {
    obj["break"] = true;
  } else if (value) {
    obj["value"] = modelica::toJSON(value.get());
  }
}

ClassModification *Modification::getClassModification() const {
  return classModification.get();
}

Expression *Modification::getValue() const { return value.get(); }

void Modification::setValue(std::unique_ptr<Expression> node) {
  value = std::move(node);
}

bool Modification::isBreak() const { return breakValue; }

void Modification::setBreak(bool isBreak) { breakValue = isBreak; }

void Modification::setClassModification(
    std::unique_ptr<ClassModification> node) {
  classModification = std::move(node);
}
} // namespace marco::ast::modelica
