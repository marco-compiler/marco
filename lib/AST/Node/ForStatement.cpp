#include "marco/AST/Node/ForStatement.h"
#include "marco/AST/Node/Induction.h"

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
    setInduction(other.induction->clone());
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
    result["induction"] = getInduction()->toJSON();

    llvm::SmallVector<llvm::json::Value> statementsJson;

    for (const auto& statement : statements) {
      statementsJson.push_back(statement->toJSON());
    }

    addJSONProperties(result);
    return result;
  }

  Induction* ForStatement::getInduction()
  {
    assert(induction != nullptr && "Induction not set");
    return induction->cast<Induction>();
  }

  const Induction* ForStatement::getInduction() const
  {
    assert(induction != nullptr && "Induction not set");
    return induction->cast<Induction>();
  }

  void ForStatement::setInduction(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Induction>());
    induction = std::move(node);
    induction->setParent(this);
  }

  size_t ForStatement::size() const
  {
    return statements.size();
  }

  Statement* ForStatement::operator[](size_t index)
  {
    assert(index < statements.size());
    return statements[index]->cast<Statement>();
  }

  const Statement* ForStatement::operator[](size_t index) const
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
