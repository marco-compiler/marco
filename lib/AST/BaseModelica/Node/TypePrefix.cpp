#include "marco/AST/BaseModelica/Node/TypePrefix.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace {
std::string toString(VariabilityQualifier qualifier) {
  switch (qualifier) {
  case VariabilityQualifier::discrete:
    return "discrete";
  case VariabilityQualifier::parameter:
    return "parameter";
  case VariabilityQualifier::constant:
    return "constant";
  case VariabilityQualifier::none:
    return "none";
  }

  llvm_unreachable("Unknown variability qualifier");
  return "unknown";
}

std::string toString(IOQualifier qualifier) {
  switch (qualifier) {
  case IOQualifier::input:
    return "input";
  case IOQualifier::output:
    return "output";
  case IOQualifier::none:
    return "none";
  }

  llvm_unreachable("Unknown I/O qualifier");
  return "unknown";
}
} // namespace

namespace marco::ast::bmodelica {
TypePrefix::TypePrefix(SourceRange location)
    : ASTNode(ASTNode::Kind::TypePrefix, std::move(location)) {}

TypePrefix::TypePrefix(const TypePrefix &other) = default;

TypePrefix::~TypePrefix() = default;

std::unique_ptr<ASTNode> TypePrefix::clone() const {
  return std::make_unique<TypePrefix>(*this);
}

llvm::json::Value TypePrefix::toJSON() const {
  llvm::json::Object result;
  result["variability"] = toString(variabilityQualifier);
  result["io"] = toString(ioQualifier);

  addJSONProperties(result);
  return result;
}

void TypePrefix::setVariabilityQualifier(VariabilityQualifier qualifier) {
  variabilityQualifier = qualifier;
}

void TypePrefix::setIOQualifier(IOQualifier qualifier) {
  ioQualifier = qualifier;
}

bool TypePrefix::isDiscrete() const {
  return variabilityQualifier == VariabilityQualifier::discrete;
}

bool TypePrefix::isParameter() const {
  return variabilityQualifier == VariabilityQualifier::parameter;
}

bool TypePrefix::isConstant() const {
  return variabilityQualifier == VariabilityQualifier::constant;
}

bool TypePrefix::isInput() const { return ioQualifier == IOQualifier::input; }

bool TypePrefix::isOutput() const { return ioQualifier == IOQualifier::output; }
} // namespace marco::ast::bmodelica
