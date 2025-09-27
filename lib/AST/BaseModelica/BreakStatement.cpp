#include "marco/AST/BaseModelica/BreakStatement.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
BreakStatement::BreakStatement(SourceRange location)
    : Statement(ASTNodeKind::Statement_Break, std::move(location)) {}

BreakStatement::BreakStatement(const BreakStatement &other) = default;

BreakStatement::~BreakStatement() = default;

std::unique_ptr<ast::ASTNode> BreakStatement::clone() const {
  return std::make_unique<BreakStatement>(*this);
}

llvm::json::Value BreakStatement::toJSON() const {
  llvm::json::Object result;
  addNodeKindToJSON(*this, result);
  return result;
}
} // namespace marco::ast::bmodelica
