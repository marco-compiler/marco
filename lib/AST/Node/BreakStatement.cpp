#include "marco/AST/Node/BreakStatement.h"
#include "marco/Parser/Location.h"
#include "marco/AST/Node/Statement.h"
#include "marco/AST/Node/ASTNode.h"
#include <utility>
#include <memory>
#include <llvm/Support/JSON.h>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
BreakStatement::BreakStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_Break, std::move(location)) {}

BreakStatement::BreakStatement(const BreakStatement &other)
     = default;

BreakStatement::~BreakStatement() = default;

std::unique_ptr<ASTNode> BreakStatement::clone() const {
  return std::make_unique<BreakStatement>(*this);
}

llvm::json::Value BreakStatement::toJSON() const {
  llvm::json::Object result;
  addJSONProperties(result);
  return result;
}
} // namespace marco::ast
