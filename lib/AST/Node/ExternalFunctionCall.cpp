#include "marco/AST/Node/ExternalFunctionCall.h"
#include "marco/AST/Node/ComponentReference.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco; 
using namespace ::marco::ast; 

namespace marco::ast {
ExternalFunctionCall::ExternalFunctionCall(SourceRange location) : ASTNode(ASTNode::Kind::External_Function_Call, std::move(location)) {}

ExternalFunctionCall::ExternalFunctionCall(const ExternalFunctionCall &other) : ASTNode(other), name(other.name) {
  setComponentReference(other.componentReference->clone()); 
  setExpressions(other.expressions);
}

ExternalFunctionCall::~ExternalFunctionCall() = default; 

std::unique_ptr<ASTNode> ExternalFunctionCall::clone() const {
  return std::make_unique<ExternalFunctionCall>(*this); 
}

llvm::json::Value ExternalFunctionCall::toJSON() const {
  llvm::json::Object result; 
  result["name"] = getName();
  result["component_reference"] = getComponentReference()->toJSON(); 
  llvm::SmallVector<llvm::json::Value> expressionsJson; 
  for (const auto &expression : expressions){
    expressionsJson.push_back(expression -> toJSON());
  }
  result["expressions"] = llvm::json::Array(expressionsJson);
  addJSONProperties(result);
  return result;  
}

void ExternalFunctionCall::setName(llvm::StringRef newName) {
  name = newName.str(); 
}

llvm::StringRef ExternalFunctionCall::getName() const {
  return name; 
}

void ExternalFunctionCall::setComponentReference(std::unique_ptr<ASTNode> node) {
  assert(node->isa<ComponentReference>()); 
  componentReference = std::move(node); 
  componentReference -> setParent(this); 
}

ComponentReference *ExternalFunctionCall::getComponentReference() {
  return componentReference->cast<ComponentReference>();
}

const ComponentReference *ExternalFunctionCall::getComponentReference() const {
  return componentReference->cast<ComponentReference>();
}

void ExternalFunctionCall::setExpressions(llvm::SmallVector<std::unique_ptr<ASTNode>> newExpressions) {
  expressions.clear();
  for (const auto &expression : newExpressions) {
    assert(expression->isa<Expression>()); 
    auto &clone = expressions.emplace_back(expression->clone());
    clone->setParent(this);
  }
}

llvm::SmallVector<std::unique_ptr<ASTNode>> ExternalFunctionCall::getExpressions() const {
  return expressions; 
} 
}
