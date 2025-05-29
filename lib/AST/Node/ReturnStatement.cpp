#include "marco/AST/Node/ReturnStatement.h"
#include "marco/Parser/Location.h"
#include "marco/AST/Node/Statement.h"
#include "marco/AST/Node/ASTNode.h"
#include <utility>
#include <memory>
#include <llvm/Support/JSON.h>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ReturnStatement::ReturnStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_Return, std::move(location)) {}

ReturnStatement::ReturnStatement(const ReturnStatement &other)
     = default;

ReturnStatement::~ReturnStatement() = default;

std::unique_ptr<ASTNode> ReturnStatement::clone() const {
  return std::make_unique<ReturnStatement>(*this);
}

llvm::json::Value ReturnStatement::toJSON() const {
  llvm::json::Object result;
  addJSONProperties(result);
  return result;
}
} // namespace marco::ast
