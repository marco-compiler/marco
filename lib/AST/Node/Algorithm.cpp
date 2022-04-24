#include "marco/AST/Node/Algorithm.h"
#include "marco/AST/Node/Statement.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Algorithm::Algorithm(
      SourceRange location,
      llvm::ArrayRef<std::unique_ptr<Statement>> statements)
    : ASTNode(std::move(location))
  {
    for (const auto& statement : statements) {
      this->statements.push_back(statement->clone());
    }
  }

  Algorithm::Algorithm(const Algorithm& other)
      : ASTNode(other)
  {
    for (const auto& statement : other.statements) {
      this->statements.push_back(statement->clone());
    }
  }

  Algorithm::Algorithm(Algorithm&& other) = default;

  Algorithm::~Algorithm() = default;

  Algorithm& Algorithm::operator=(const Algorithm& other)
  {
    Algorithm result(other);
    swap(*this, result);
    return *this;
  }

  Algorithm& Algorithm::operator=(Algorithm&& other) = default;

  void swap(Algorithm& first, Algorithm& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    impl::swap(first.statements, second.statements);
  }

void Algorithm::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "algorithm\n";

    for (const auto& statement : *this) {
      statement->dump(os, indents + 1);
    }
  }

  Statement* Algorithm::operator[](size_t index)
  {
    assert(index < statements.size());
    return statements[index].get();
  }

  const Statement* Algorithm::operator[](size_t index) const
  {
    assert(index < statements.size());
    return statements[index].get();
  }

  llvm::MutableArrayRef<std::unique_ptr<Statement>> Algorithm::getBody()
  {
    return statements;
  }

  llvm::ArrayRef<std::unique_ptr<Statement>> Algorithm::getBody() const
  {
    return statements;
  }

  void Algorithm::setBody(llvm::ArrayRef<std::unique_ptr<Statement>> body)
  {
    statements.clear();

    for (const auto& statement : body) {
      statements.push_back(statement->clone());
    }
  }

  size_t Algorithm::size() const
  {
    return statements.size();
  }

  Algorithm::statements_iterator Algorithm::begin()
  {
    return statements.begin();
  }

  Algorithm::statements_const_iterator Algorithm::begin() const
  {
    return statements.begin();
  }

  Algorithm::statements_iterator Algorithm::end()
  {
    return statements.end();
  }

  Algorithm::statements_const_iterator Algorithm::end() const
  {
    return statements.end();
  }
}
