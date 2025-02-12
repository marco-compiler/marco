#include "marco/AST/Node/WhenStatement.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
WhenStatement::WhenStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_When, std::move(location)) {}

WhenStatement::WhenStatement(const WhenStatement &other) : Statement(other) {
  setCondition(other.condition->clone());
  setStatements(other.statements);
}

WhenStatement::~WhenStatement() = default;

std::unique_ptr<ASTNode> WhenStatement::clone() const {
  return std::make_unique<WhenStatement>(*this);
}

llvm::json::Value WhenStatement::toJSON() const {
  llvm::json::Object result;

  llvm::SmallVector<llvm::json::Value> statementsJson;

  for (const auto &statement : statements) {
    statementsJson.push_back(statement->toJSON());
  }

  result["statements"] = llvm::json::Array(statementsJson);

  addJSONProperties(result);
  return result;
}

Expression *WhenStatement::getCondition() {
  assert(condition != nullptr && "Condition not set");
  return condition->cast<Expression>();
}

const Expression *WhenStatement::getCondition() const {
  assert(condition != nullptr && "Condition not set");
  return condition->cast<Expression>();
}

void WhenStatement::setCondition(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  condition = std::move(node);
  condition->setParent(this);
}

size_t WhenStatement::size() const { return statements.size(); }

Statement *WhenStatement::operator[](size_t index) {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

const Statement *WhenStatement::operator[](size_t index) const {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> WhenStatement::getStatements() const {
  return statements;
}

void WhenStatement::setStatements(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  statements.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Statement>());
    auto &clone = statements.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
