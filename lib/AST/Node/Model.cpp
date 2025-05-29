#include "marco/AST/Node/Model.h"
#include "marco/Parser/Location.h"
#include "marco/AST/Node/Class.h"
#include "marco/AST/Node/ASTNode.h"
#include <utility>
#include <memory>
#include <llvm/Support/JSON.h>

using namespace marco::ast;

namespace marco::ast {
Model::Model(SourceRange location)
    : Class(ASTNode::Kind::Class_Model, std::move(location)) {}

std::unique_ptr<ASTNode> Model::clone() const {
  return std::make_unique<Model>(*this);
}

llvm::json::Value Model::toJSON() const {
  llvm::json::Object result;
  addJSONProperties(result);
  return result;
}
} // namespace marco::ast
