#include "marco/AST/Modelica/Modification.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Modification::setClassModification(
    std::unique_ptr<ClassModification> node) {
  classModification = std::move(node);
}

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

void ClassModification::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(arguments, callback);
}

void ClassModification::addJSONProperties(llvm::json::Object &obj) const {
  obj["arguments"] = modelica::toJSON(arguments);
}

void ElementModification::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(modification, callback);
}

void ElementModification::addJSONProperties(llvm::json::Object &obj) const {
  if (each) {
    obj["each"] = true;
  }

  if (final) {
    obj["final"] = true;
  }

  obj["name"] = modelica::toJSON(name);

  if (modification) {
    obj["modification"] = modelica::toJSON(modification.get());
  }

  if (!description.empty()) {
    obj["description"] = description;
  }
}

void ConstrainingClause::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(classModification, callback);
  walk(annotation, callback);
}

void ConstrainingClause::addJSONProperties(llvm::json::Object &obj) const {
  obj["type"] = modelica::toJSON(typeSpecifier);

  if (classModification) {
    obj["class_modification"] = modelica::toJSON(classModification.get());
  }

  if (!description.empty()) {
    obj["description"] = description;
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

void ElementRedeclaration::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(element, callback);
  walk(constrainingClause, callback);
}

void ElementRedeclaration::addJSONProperties(llvm::json::Object &obj) const {
  obj["redeclare"] = redeclare;

  if (each) {
    obj["each"] = true;
  }

  if (final) {
    obj["final"] = true;
  }

  if (replaceable) {
    obj["replaceable"] = true;
  }

  obj["element"] = modelica::toJSON(element.get());

  if (constrainingClause) {
    obj["constraining_clause"] = modelica::toJSON(constrainingClause.get());
  }
}

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

void Description::addToJSON(llvm::json::Object &obj) const {
  if (!text.empty()) {
    obj["description"] = text;
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}
} // namespace marco::ast::modelica
