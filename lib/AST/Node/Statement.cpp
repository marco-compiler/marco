#include "marco/AST/Node/Statement.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Induction.h"
#include "marco/AST/Node/Tuple.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Statement::Statement(const Statement& other)
      : ASTNode(other)
  {
  }

  Statement::~Statement() = default;
}
