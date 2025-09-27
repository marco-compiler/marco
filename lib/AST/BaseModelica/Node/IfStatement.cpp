#include "marco/AST/BaseModelica/Node/IfStatement.h"
#include "marco/AST/BaseModelica/Node/Expression.h"
#include "marco/AST/BaseModelica/Node/StatementsBlock.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
IfStatement::IfStatement(SourceRange location)
    : ASTNode(ASTNode::Kind::Statement_If, std::move(location)) {}

IfStatement::IfStatement(const IfStatement &other) : ASTNode(other) {
  setIfCondition(other.ifCondition->clone());
  setIfBlock(other.ifBlock->clone());
  setElseIfConditions(other.elseIfConditions);
  setElseIfBlocks(other.elseIfBlocks);

  if (other.hasElseBlock()) {
    setElseBlock(other.elseBlock->clone());
  }
}

IfStatement::~IfStatement() = default;

std::unique_ptr<ASTNode> IfStatement::clone() const {
  return std::make_unique<IfStatement>(*this);
}

llvm::json::Value IfStatement::toJSON() const {
  llvm::json::Object result;
  result["if_condition"] = getIfCondition()->toJSON();
  result["if_block"] = getIfBlock()->toJSON();

  llvm::SmallVector<llvm::json::Value> elseIfConditionsJson;

  for (const auto &condition : elseIfConditions) {
    elseIfConditionsJson.push_back(condition->toJSON());
  }

  llvm::SmallVector<llvm::json::Value> elseIfBlocksJson;

  for (const auto &block : elseIfBlocks) {
    elseIfBlocksJson.push_back(block->toJSON());
  }

  if (hasElseBlock()) {
    result["else_block"] = getElseBlock()->toJSON();
  }

  addJSONProperties(result);
  return result;
}

Expression *IfStatement::getIfCondition() {
  assert(ifCondition != nullptr && "If condition not set");
  return ifCondition->cast<Expression>();
}

const Expression *IfStatement::getIfCondition() const {
  assert(ifCondition != nullptr && "If condition not set");
  return ifCondition->cast<Expression>();
}

void IfStatement::setIfCondition(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  ifCondition = std::move(node);
  ifCondition->setParent(this);
}

StatementsBlock *IfStatement::getIfBlock() {
  assert(ifBlock != nullptr && "If block not set");
  return ifBlock->cast<StatementsBlock>();
}

const StatementsBlock *IfStatement::getIfBlock() const {
  assert(ifBlock != nullptr && "If block not set");
  return ifBlock->cast<StatementsBlock>();
}

void IfStatement::setIfBlock(std::unique_ptr<ASTNode> node) {
  assert(node->isa<StatementsBlock>());
  ifBlock = std::move(node);
  ifBlock->setParent(this);
}

size_t IfStatement::getNumOfElseIfBlocks() const { return elseIfBlocks.size(); }

bool IfStatement::hasElseIfBlocks() const { return !elseIfBlocks.empty(); }

Expression *IfStatement::getElseIfCondition(size_t index) {
  assert(index < elseIfConditions.size());
  return elseIfConditions[index]->cast<Expression>();
}

const Expression *IfStatement::getElseIfCondition(size_t index) const {
  assert(index < elseIfConditions.size());
  return elseIfConditions[index]->cast<Expression>();
}

void IfStatement::setElseIfConditions(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseIfConditions.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = elseIfConditions.emplace_back(node->clone());
    clone->setParent(this);
  }
}

StatementsBlock *IfStatement::getElseIfBlock(size_t index) {
  assert(index < elseIfBlocks.size());
  return elseIfBlocks[index]->cast<StatementsBlock>();
}

const StatementsBlock *IfStatement::getElseIfBlock(size_t index) const {
  assert(index < elseIfBlocks.size());
  return elseIfBlocks[index]->cast<StatementsBlock>();
}

void IfStatement::setElseIfBlocks(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseIfBlocks.clear();

  for (const auto &node : nodes) {
    assert(node->isa<StatementsBlock>());
    auto &clone = elseIfBlocks.emplace_back(node->clone());
    clone->setParent(this);
  }
}

bool IfStatement::hasElseBlock() const { return elseBlock != nullptr; }

StatementsBlock *IfStatement::getElseBlock() {
  assert(elseBlock != nullptr && "Else block not set");
  return elseBlock->cast<StatementsBlock>();
}

const StatementsBlock *IfStatement::getElseBlock() const {
  assert(elseBlock != nullptr && "Else block not set");
  return elseBlock->cast<StatementsBlock>();
}

void IfStatement::setElseBlock(std::unique_ptr<ASTNode> node) {
  assert(node->isa<StatementsBlock>());
  elseBlock = std::move(node);
  elseBlock->setParent(this);
}
} // namespace marco::ast::bmodelica
