#include "marco/AST/BaseModelica/Node/ReturnStatement.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
ReturnStatement::ReturnStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_Return, std::move(location)) {}

ReturnStatement::ReturnStatement(const ReturnStatement &other) = default;

ReturnStatement::~ReturnStatement() = default;

std::unique_ptr<ASTNode> ReturnStatement::clone() const {
  return std::make_unique<ReturnStatement>(*this);
}

llvm::json::Value ReturnStatement::toJSON() const {
  llvm::json::Object result;
  addJSONProperties(result);
  return result;
}
} // namespace marco::ast::bmodelica
