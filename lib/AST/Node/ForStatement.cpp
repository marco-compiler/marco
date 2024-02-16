#include "marco/AST/Node/ForStatement.h"
#include "marco/AST/Node/ForIndex.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ForStatement::ForStatement(SourceRange location)
      : Statement(ASTNode::Kind::Statement_For, std::move(location))
  {
  }

  ForStatement::ForStatement(const ForStatement& other)
      : Statement(other)
  {
    setForIndices(other.forIndices);
    setStatements(other.statements);
  }

  ForStatement::~ForStatement() = default;

  std::unique_ptr<ASTNode> ForStatement::clone() const
  {
    return std::make_unique<ForStatement>(*this);
  }

  llvm::json::Value ForStatement::toJSON() const
  {
    llvm::json::Object result;
    llvm::SmallVector<llvm::json::Value> forIndicesJson;

    for (const auto& forIndex : forIndices) {
      forIndicesJson.push_back(forIndex->toJSON());
    }
    
    result["for_indices"] = llvm::json::Array(forIndicesJson);
    llvm::SmallVector<llvm::json::Value> statementsJson;

    for (const auto& statement : statements) {
      statementsJson.push_back(statement->toJSON());
    }

    result["statements"] = llvm::json::Array(statementsJson);

    addJSONProperties(result);
    return result;
  }

  size_t ForStatement::getNumOfForIndices() const
  {
    return forIndices.size();
  }

  ForIndex* ForStatement::getForIndex(size_t index)
  {
    assert(index < forIndices.size());
    return forIndices[index]->cast<ForIndex>();
  }

  const ForIndex* ForStatement::getForIndex(size_t index) const
  {
    assert(index < forIndices.size());
    return forIndices[index]->cast<ForIndex>();
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> ForStatement::getForIndices() const
  {
    return forIndices;
  }

  void ForStatement::setForIndices(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    forIndices.clear();

    for (const auto& node : nodes) {
      assert(node->isa<ForIndex>());
      auto& clone = forIndices.emplace_back(node->clone());
      clone->setParent(this);
    }
  }

  size_t ForStatement::getNumOfStatements() const
  {
    return statements.size();
  }

  Statement* ForStatement::getStatement(size_t index)
  {
    assert(index < statements.size());
    return statements[index]->cast<Statement>();
  }

  const Statement* ForStatement::getStatement(size_t index) const
  {
    assert(index < statements.size());
    return statements[index]->cast<Statement>();
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> ForStatement::getStatements() const
  {
    return statements;
  }

  void ForStatement::setStatements(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    statements.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Statement>());
      auto& clone = statements.emplace_back(node->clone());
      clone->setParent(this);
    }
  }
}
