#include "marco/AST/BaseModelica/StatementsBlock.h"
#include "marco/AST/BaseModelica/Statement.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
StatementsBlock::StatementsBlock(SourceRange location)
    : ASTNode(ASTNodeKind::StatementsBlock, std::move(location)) {}

StatementsBlock::StatementsBlock(const StatementsBlock &other)
    : ASTNode(other) {
  setBody(other.statements);
}

StatementsBlock::~StatementsBlock() = default;

std::unique_ptr<ASTNode> StatementsBlock::clone() const {
  return std::make_unique<StatementsBlock>(*this);
}

llvm::json::Value StatementsBlock::toJSON() const {
  llvm::json::Object result;

  llvm::SmallVector<llvm::json::Value> statementsJson;

  for (const auto &statement : statements) {
    statementsJson.push_back(statement->toJSON());
  }

  result["statements"] = llvm::json::Array(statementsJson);

  addNodeKindToJSON(*this, result);
  return result;
}

size_t StatementsBlock::size() const { return statements.size(); }

Statement *StatementsBlock::operator[](size_t index) {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

const Statement *StatementsBlock::operator[](size_t index) const {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

void StatementsBlock::setBody(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  statements.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Statement>());
    auto &clone = statements.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
