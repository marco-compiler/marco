#include "march/AST/Node/ExternalFunctionCall.h"



ExternalFunctionCall(SourceRange location) : ASTNode(ASTNode::Kind::External_Function_Call, std::move(location)) {}

~ExternalFunctionCall() = default; 

std::unique_ptr<ASTNode> clone() const {
  return std::make_unique<ExternalFunctionCall>(*this); 
}

llvm::json::Value toJSON() const {
  llvm::json::Object result; 
  result['name'] = getName();
  result['component_reference'] = getComponentReferencePtr().toJSON(); 
  llvm::SmallVector<llvm::json::Value> expressionsJson; 
  for (const auto &expression : expressions){
    expressionsJson.push_back(expression -> toJSON());
  }
  result['expressions'] = llvm::json::Array(expressionsJson);
  addJSONProperties(result);
  return result;  
}

void setName(llvm::StringRef name) {
  name = name.str(); 
}

llvm::StringRef getName() const {
  return name; 
}

void setComponentReference(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression_ComponentReference>()); 
  componentReference = std::move(node); 
  componentReference -> setParent(this); 
}

std::unique_ptr<ASTNode> getComponentReference() {
  return componentReference; 
}

ComponentReference *getComponentReferencePtr() {
  return componentReference->cast<ComponentReference>();
}

void setExpressions(llvm::ArrayRef<std::unique_ptr<ASTNode>> newExpressions) {
  expressions.clear();
  for (const auto &expression : newExpressions) {
    assert(expression->isa<Expression>()); 
    auto &clone = expressions.emplace_back(expression->clone());
    clone->setParent(this);
  }
}

llvm::ArrayRef<std::unique_ptr> getExpressions(){
  return expressions; 
} 
