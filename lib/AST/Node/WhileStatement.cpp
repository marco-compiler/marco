#include "marco/AST/Node/WhileStatement.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
WhileStatement::WhileStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_While, std::move(location)) {}

WhileStatement::WhileStatement(const WhileStatement &other) : Statement(other) {
  setCondition(other.condition->clone());
  setStatements(other.statements);
}

WhileStatement::~WhileStatement() = default;

std::unique_ptr<ASTNode> WhileStatement::clone() const {
  return std::make_unique<WhileStatement>(*this);
}

llvm::json::Value WhileStatement::toJSON() const {
  llvm::json::Object result;

  llvm::SmallVector<llvm::json::Value> statementsJson;

  for (const auto &statement : statements) {
    statementsJson.push_back(statement->toJSON());
  }

  result["statements"] = llvm::json::Array(statementsJson);

  addJSONProperties(result);
  return result;
}

Expression *WhileStatement::getCondition() {
  assert(condition != nullptr && "Condition not set");
  return condition->cast<Expression>();
}

const Expression *WhileStatement::getCondition() const {
  assert(condition != nullptr && "Condition not set");
  return condition->cast<Expression>();
}

void WhileStatement::setCondition(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  condition = std::move(node);
  condition->setParent(this);
}

size_t WhileStatement::size() const { return statements.size(); }

Statement *WhileStatement::operator[](size_t index) {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

const Statement *WhileStatement::operator[](size_t index) const {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> WhileStatement::getStatements() const {
  return statements;
}

void WhileStatement::setStatements(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  statements.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Statement>());
    auto &clone = statements.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
