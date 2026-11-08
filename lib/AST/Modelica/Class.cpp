#include "marco/AST/Modelica/Class.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
//===---------------------------------------------------------------------===//
// Prefixes
//===---------------------------------------------------------------------===//

void ElementPrefixes::addToJSON(llvm::json::Object &obj) const {
  auto addFlag = [&](llvm::StringRef name, bool value) {
    if (value) {
      obj[name] = true;
    }
  };

  addFlag("protected", isProtected);
  addFlag("redeclare", isRedeclare);
  addFlag("final", isFinal);
  addFlag("inner", isInner);
  addFlag("outer", isOuter);
  addFlag("replaceable", isReplaceable);
}

void TypePrefix::addToJSON(llvm::json::Object &obj) const {
  if (connectorKind != ConnectorKind::None) {
    obj["connector_kind"] = toString(connectorKind);
  }

  if (variability != Variability::None) {
    obj["variability"] = toString(variability);
  }

  if (causality != Causality::None) {
    obj["causality"] = toString(causality);
  }
}

llvm::StringRef toString(ConnectorKind kind) {
  switch (kind) {
  case ConnectorKind::None:
    return "";
  case ConnectorKind::Flow:
    return "flow";
  case ConnectorKind::Stream:
    return "stream";
  }

  llvm_unreachable("Unknown connector kind");
  return "";
}

llvm::StringRef toString(Variability variability) {
  switch (variability) {
  case Variability::None:
    return "";
  case Variability::Discrete:
    return "discrete";
  case Variability::Parameter:
    return "parameter";
  case Variability::Constant:
    return "constant";
  }

  llvm_unreachable("Unknown variability");
  return "";
}

llvm::StringRef toString(Causality causality) {
  switch (causality) {
  case Causality::None:
    return "";
  case Causality::Input:
    return "input";
  case Causality::Output:
    return "output";
  }

  llvm_unreachable("Unknown causality");
  return "";
}

llvm::StringRef toString(ClassRestriction restriction) {
  switch (restriction) {
  case ClassRestriction::Class:
    return "class";
  case ClassRestriction::Model:
    return "model";
  case ClassRestriction::Record:
    return "record";
  case ClassRestriction::OperatorRecord:
    return "operator record";
  case ClassRestriction::Block:
    return "block";
  case ClassRestriction::Connector:
    return "connector";
  case ClassRestriction::ExpandableConnector:
    return "expandable connector";
  case ClassRestriction::Type:
    return "type";
  case ClassRestriction::Package:
    return "package";
  case ClassRestriction::Function:
    return "function";
  case ClassRestriction::Operator:
    return "operator";
  case ClassRestriction::OperatorFunction:
    return "operator function";
  }

  llvm_unreachable("Unknown class restriction");
  return "";
}

//===---------------------------------------------------------------------===//
// EnumerationLiteral
//===---------------------------------------------------------------------===//

void EnumerationLiteral::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void EnumerationLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  description.addToJSON(obj);
}

//===---------------------------------------------------------------------===//
// ClassDefinition
//===---------------------------------------------------------------------===//

void ClassDefinition::setExternalClause(std::unique_ptr<ExternalClause> node) {
  externalClause = std::move(node);
}

void ClassDefinition::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
  walk(elements, callback);
  walk(externalClause, callback);
  walk(classModification, callback);
  walk(subscripts, callback);
  walk(enumerationLiterals, callback);
  walk(constrainingClause, callback);
}

void ClassDefinition::addJSONProperties(llvm::json::Object &obj) const {
  prefixes.addToJSON(obj);

  if (encapsulated) {
    obj["encapsulated"] = true;
  }

  if (partial) {
    obj["partial"] = true;
  }

  if (purity == Purity::Pure) {
    obj["purity"] = "pure";
  } else if (purity == Purity::Impure) {
    obj["purity"] = "impure";
  }

  obj["restriction"] = toString(restriction);
  obj["name"] = name;
  description.addToJSON(obj);

  switch (form) {
  case Form::Long:
  case Form::ClassExtends:
    obj["form"] = form == Form::Long ? "long" : "class_extends";

    if (classModification) {
      obj["class_modification"] = modelica::toJSON(classModification.get());
    }

    obj["elements"] = modelica::toJSON(elements);

    if (externalClause) {
      obj["external"] = modelica::toJSON(externalClause.get());
    }

    break;

  case Form::Short:
    obj["form"] = "short";

    if (basePrefix != Causality::None) {
      obj["base_prefix"] = toString(basePrefix);
    }

    obj["base_class"] = modelica::toJSON(baseClass);

    if (!subscripts.empty()) {
      obj["subscripts"] = modelica::toJSON(subscripts);
    }

    if (classModification) {
      obj["class_modification"] = modelica::toJSON(classModification.get());
    }

    break;

  case Form::Enumeration:
    obj["form"] = "enumeration";

    if (unspecifiedEnumeration) {
      obj["unspecified"] = true;
    } else {
      obj["literals"] = modelica::toJSON(enumerationLiterals);
    }

    break;

  case Form::Derivative: {
    obj["form"] = "derivative";
    obj["function"] = modelica::toJSON(derivedFunction);
    llvm::json::Array args;

    for (const std::string &arg : derivativeArguments) {
      args.push_back(arg);
    }

    obj["arguments"] = std::move(args);
    break;
  }
  }

  if (constrainingClause) {
    obj["constraining_clause"] = modelica::toJSON(constrainingClause.get());
  }
}

//===---------------------------------------------------------------------===//
// Components
//===---------------------------------------------------------------------===//

void ComponentDeclaration::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(subscripts, callback);
  walk(modification, callback);
  walk(condition, callback);
  description.walkChildren(callback);
}

void ComponentDeclaration::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;

  if (!subscripts.empty()) {
    obj["subscripts"] = modelica::toJSON(subscripts);
  }

  if (modification) {
    obj["modification"] = modelica::toJSON(modification.get());
  }

  if (condition) {
    obj["condition"] = modelica::toJSON(condition.get());
  }

  description.addToJSON(obj);
}

void ComponentClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(subscripts, callback);
  walk(declarations, callback);
  walk(constrainingClause, callback);
}

void ComponentClause::addJSONProperties(llvm::json::Object &obj) const {
  prefixes.addToJSON(obj);
  typePrefix.addToJSON(obj);
  obj["type"] = modelica::toJSON(typeSpecifier);

  if (!subscripts.empty()) {
    obj["subscripts"] = modelica::toJSON(subscripts);
  }

  obj["declarations"] = modelica::toJSON(declarations);

  if (constrainingClause) {
    obj["constraining_clause"] = modelica::toJSON(constrainingClause.get());
  }
}

//===---------------------------------------------------------------------===//
// Extends and import clauses
//===---------------------------------------------------------------------===//

void ExtendsClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(classModification, callback);
  walk(annotation, callback);
}

void ExtendsClause::addJSONProperties(llvm::json::Object &obj) const {
  if (protectedElement) {
    obj["protected"] = true;
  }

  obj["base_class"] = modelica::toJSON(baseClass);

  if (classModification) {
    obj["class_modification"] = modelica::toJSON(classModification.get());
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

void ImportClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void ImportClause::addJSONProperties(llvm::json::Object &obj) const {
  if (protectedElement) {
    obj["protected"] = true;
  }

  obj["name"] = modelica::toJSON(name);

  switch (form) {
  case Form::Qualified:
    obj["form"] = "qualified";
    break;

  case Form::Renaming:
    obj["form"] = "renaming";
    obj["alias"] = alias;
    break;

  case Form::Unqualified:
    obj["form"] = "unqualified";
    break;

  case Form::Multiple: {
    obj["form"] = "multiple";
    llvm::json::Array names;

    for (const std::string &importedName : importedNames) {
      names.push_back(importedName);
    }

    obj["names"] = std::move(names);
    break;
  }
  }

  description.addToJSON(obj);
}

//===---------------------------------------------------------------------===//
// Sections
//===---------------------------------------------------------------------===//

void EquationSection::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(equations, callback);
}

void EquationSection::addJSONProperties(llvm::json::Object &obj) const {
  obj["initial"] = initial;
  obj["equations"] = modelica::toJSON(equations);
}

void AlgorithmSection::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(statements, callback);
}

void AlgorithmSection::addJSONProperties(llvm::json::Object &obj) const {
  obj["initial"] = initial;
  obj["statements"] = modelica::toJSON(statements);
}

void ExternalClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(output, callback);
  walk(arguments, callback);
  walk(annotation, callback);
}

void ExternalClause::addJSONProperties(llvm::json::Object &obj) const {
  if (language) {
    obj["language"] = *language;
  }

  if (hasFunctionCall()) {
    obj["function"] = function;
    obj["arguments"] = modelica::toJSON(arguments);

    if (output) {
      obj["output"] = modelica::toJSON(output.get());
    }
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

//===---------------------------------------------------------------------===//
// StoredDefinition
//===---------------------------------------------------------------------===//

void StoredDefinition::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(classes, callback);
}

void StoredDefinition::addJSONProperties(llvm::json::Object &obj) const {
  if (within) {
    obj["within"] = modelica::toJSON(*within);
  }

  obj["classes"] = modelica::toJSON(classes);
}
} // namespace marco::ast::modelica
