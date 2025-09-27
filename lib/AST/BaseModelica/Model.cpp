#include "marco/AST/BaseModelica/Model.h"

using namespace marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Model::Model(SourceRange location)
    : Class(ASTNodeKind::Class_Model, std::move(location)) {}

std::unique_ptr<ASTNode> Model::clone() const {
  return std::make_unique<Model>(*this);
}

llvm::json::Value Model::toJSON() const {
  llvm::json::Object result;
  addNodeKindToJSON(*this, result);
  return result;
}
} // namespace marco::ast::bmodelica
