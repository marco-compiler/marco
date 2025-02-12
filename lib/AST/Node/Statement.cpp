#include "marco/AST/Node/Statement.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
Statement::Statement(const Statement &other) : ASTNode(other) {}

Statement::~Statement() = default;
} // namespace marco::ast
