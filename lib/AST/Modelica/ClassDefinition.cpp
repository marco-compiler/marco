#include "marco/AST/Modelica/ClassDefinition.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
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

ElementPrefixes &ClassDefinition::getPrefixes() { return prefixes; }

const ElementPrefixes &ClassDefinition::getPrefixes() const { return prefixes; }

bool ClassDefinition::isEncapsulated() const { return encapsulated; }

void ClassDefinition::setEncapsulated(bool value) { encapsulated = value; }

bool ClassDefinition::isPartial() const { return partial; }

void ClassDefinition::setPartial(bool value) { partial = value; }

Purity ClassDefinition::getPurity() const { return purity; }

void ClassDefinition::setPurity(Purity value) { purity = value; }

ClassRestriction ClassDefinition::getRestriction() const { return restriction; }

void ClassDefinition::setRestriction(ClassRestriction value) {
  restriction = value;
}

llvm::StringRef ClassDefinition::getName() const { return name; }

void ClassDefinition::setName(std::string newName) {
  name = std::move(newName);
}

ClassDefinition::Form ClassDefinition::getForm() const { return form; }

void ClassDefinition::setForm(Form value) { form = value; }

Description &ClassDefinition::getDescription() { return description; }

const Description &ClassDefinition::getDescription() const {
  return description;
}

ChildrenRef<Node> ClassDefinition::getElements() const { return elements; }

void ClassDefinition::addElement(std::unique_ptr<Node> node) {
  elements.emplace_back(std::move(node));
}

ExternalClause *ClassDefinition::getExternalClause() const {
  return externalClause.get();
}

ClassModification *ClassDefinition::getClassModification() const {
  return classModification.get();
}

void ClassDefinition::setClassModification(
    std::unique_ptr<ClassModification> node) {
  classModification = std::move(node);
}

Causality ClassDefinition::getBasePrefix() const { return basePrefix; }

void ClassDefinition::setBasePrefix(Causality value) { basePrefix = value; }

const Name &ClassDefinition::getBaseClass() const { return baseClass; }

void ClassDefinition::setBaseClass(Name name) { baseClass = std::move(name); }

ChildrenRef<Expression> ClassDefinition::getSubscripts() const {
  return subscripts;
}

void ClassDefinition::addSubscript(std::unique_ptr<Expression> node) {
  subscripts.emplace_back(std::move(node));
}

ChildrenRef<EnumerationLiteral>
ClassDefinition::getEnumerationLiterals() const {
  return enumerationLiterals;
}

void ClassDefinition::addEnumerationLiteral(
    std::unique_ptr<EnumerationLiteral> node) {
  enumerationLiterals.emplace_back(std::move(node));
}

bool ClassDefinition::isUnspecifiedEnumeration() const {
  return unspecifiedEnumeration;
}

void ClassDefinition::setUnspecifiedEnumeration(bool value) {
  unspecifiedEnumeration = value;
}

const Name &ClassDefinition::getDerivedFunction() const {
  return derivedFunction;
}

void ClassDefinition::setDerivedFunction(Name name) {
  derivedFunction = std::move(name);
}

llvm::ArrayRef<std::string> ClassDefinition::getDerivativeArguments() const {
  return derivativeArguments;
}

void ClassDefinition::addDerivativeArgument(std::string arg) {
  derivativeArguments.push_back(std::move(arg));
}

ConstrainingClause *ClassDefinition::getConstrainingClause() const {
  return constrainingClause.get();
}

void ClassDefinition::setConstrainingClause(
    std::unique_ptr<ConstrainingClause> node) {
  constrainingClause = std::move(node);
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

void ClassDefinition::setExternalClause(std::unique_ptr<ExternalClause> node) {
  externalClause = std::move(node);
}
} // namespace marco::ast::modelica
