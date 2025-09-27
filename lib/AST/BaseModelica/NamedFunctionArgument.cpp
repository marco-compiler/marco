#include "marco/AST/BaseModelica/NamedFunctionArgument.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
NamedFunctionArgument::NamedFunctionArgument(SourceRange location)
    : FunctionArgument(ASTNodeKind::FunctionArgument_Named,
                       std::move(location)) {}

NamedFunctionArgument::NamedFunctionArgument(const NamedFunctionArgument &other)
    : FunctionArgument(other), name(other.name) {
  setValue(other.value->clone());
}

NamedFunctionArgument::~NamedFunctionArgument() = default;

std::unique_ptr<ASTNode> NamedFunctionArgument::clone() const {
  return std::make_unique<NamedFunctionArgument>(*this);
}

llvm::json::Value NamedFunctionArgument::toJSON() const {
  llvm::json::Object result;
  result["name"] = name;
  result["value"] = getValue()->toJSON();

  addNodeKindToJSON(*this, result);
  return result;
}

llvm::StringRef NamedFunctionArgument::getName() const { return name; }

void NamedFunctionArgument::setName(llvm::StringRef newName) {
  name = newName.str();
}

FunctionArgument *NamedFunctionArgument::getValue() {
  assert(value && "Value not set");
  return value->cast<FunctionArgument>();
}

const FunctionArgument *NamedFunctionArgument::getValue() const {
  assert(value && "Value not set");
  return value->cast<FunctionArgument>();
}

void NamedFunctionArgument::setValue(std::unique_ptr<ASTNode> node) {
  assert(node->isa<FunctionArgument>());
  value = std::move(node);
  value->setParent(this);
}
} // namespace marco::ast::bmodelica
