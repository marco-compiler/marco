#include "marco/AST/Modelica/Node.h"
#include "llvm/ADT/StringExtras.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
llvm::StringRef toString(NodeKind kind) {
  switch (kind) {
  case NodeKind::StoredDefinition:
    return "stored_definition";
  case NodeKind::ClassDefinition:
    return "class_definition";
  case NodeKind::ComponentClause:
    return "component_clause";
  case NodeKind::ComponentDeclaration:
    return "component_declaration";
  case NodeKind::ExtendsClause:
    return "extends_clause";
  case NodeKind::ImportClause:
    return "import_clause";
  case NodeKind::EquationSection:
    return "equation_section";
  case NodeKind::AlgorithmSection:
    return "algorithm_section";
  case NodeKind::ExternalClause:
    return "external_clause";
  case NodeKind::EnumerationLiteral:
    return "enumeration_literal";
  case NodeKind::ConstrainingClause:
    return "constraining_clause";
  case NodeKind::Modification:
    return "modification";
  case NodeKind::ClassModification:
    return "class_modification";
  case NodeKind::Argument_ElementModification:
    return "element_modification";
  case NodeKind::Argument_ElementRedeclaration:
    return "element_redeclaration";
  case NodeKind::Argument_InheritanceModification:
    return "inheritance_modification";
  case NodeKind::Equation_Equality:
    return "equality_equation";
  case NodeKind::Equation_Call:
    return "call_equation";
  case NodeKind::Equation_Connect:
    return "connect_equation";
  case NodeKind::Equation_If:
    return "if_equation";
  case NodeKind::Equation_For:
    return "for_equation";
  case NodeKind::Equation_When:
    return "when_equation";
  case NodeKind::Statement_Assignment:
    return "assignment_statement";
  case NodeKind::Statement_MultipleAssignment:
    return "multiple_assignment_statement";
  case NodeKind::Statement_Call:
    return "call_statement";
  case NodeKind::Statement_Break:
    return "break_statement";
  case NodeKind::Statement_Return:
    return "return_statement";
  case NodeKind::Statement_If:
    return "if_statement";
  case NodeKind::Statement_For:
    return "for_statement";
  case NodeKind::Statement_While:
    return "while_statement";
  case NodeKind::Statement_When:
    return "when_statement";
  case NodeKind::Expression_IntegerLiteral:
    return "integer";
  case NodeKind::Expression_RealLiteral:
    return "real";
  case NodeKind::Expression_StringLiteral:
    return "string";
  case NodeKind::Expression_BooleanLiteral:
    return "boolean";
  case NodeKind::Expression_Time:
    return "time";
  case NodeKind::Expression_End:
    return "end";
  case NodeKind::Expression_Colon:
    return "colon";
  case NodeKind::Expression_ComponentReference:
    return "component_reference";
  case NodeKind::Expression_UnaryOperation:
    return "unary_operation";
  case NodeKind::Expression_BinaryOperation:
    return "binary_operation";
  case NodeKind::Expression_Range:
    return "range";
  case NodeKind::Expression_If:
    return "if_expression";
  case NodeKind::Expression_Call:
    return "call";
  case NodeKind::Expression_ArrayConstructor:
    return "array_constructor";
  case NodeKind::Expression_ArrayConcatenation:
    return "array_concatenation";
  case NodeKind::Expression_OutputList:
    return "output_list";
  case NodeKind::Expression_Subscripted:
    return "subscripted";
  case NodeKind::Expression_MemberAccess:
    return "member_access";
  case NodeKind::Expression_FunctionPartialApplication:
    return "function_partial_application";
  case NodeKind::ForIndex:
    return "for_index";
  case NodeKind::NamedArgument:
    return "named_argument";
  case NodeKind::Argument:
  case NodeKind::Argument_Last:
  case NodeKind::Equation:
  case NodeKind::Equation_Last:
  case NodeKind::Statement:
  case NodeKind::Statement_Last:
  case NodeKind::Expression:
  case NodeKind::Expression_Last:
    break;
  }

  llvm_unreachable("Abstract node kind");
  return "";
}

std::string Name::toString() const {
  std::string result = global ? "." : "";
  result += llvm::join(parts, ".");
  return result;
}

void Node::setParentOfChildren() {
  walkChildren([&](Node *child) {
    child->setParent(this);
    child->setParentOfChildren();
  });
}

llvm::json::Value Node::toJSON() const {
  llvm::json::Object obj;
  obj["kind"] = toString(getNodeKind());
  addJSONProperties(obj);
  return obj;
}

void Node::addJSONProperties(llvm::json::Object &obj) const {}

llvm::json::Value toJSON(const Name &name) { return name.toString(); }

llvm::json::Value toJSON(const Node *node) {
  if (!node) {
    return nullptr;
  }

  return node->toJSON();
}
} // namespace marco::ast::modelica
