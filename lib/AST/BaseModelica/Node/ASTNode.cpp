#include "marco/AST/BaseModelica/Node/ASTNode.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace {
std::string toString(ASTNode::Kind kind) {
  switch (kind) {
  case ASTNode::Kind::Root:
    return "root";
  case ASTNode::Kind::Algorithm:
    return "algorithm";
  case ASTNode::Kind::Annotation:
    return "annotation";
  case ASTNode::Kind::Argument_ElementModification:
    return "element_modification";
  case ASTNode::Kind::Argument_ElementRedeclaration:
    return "element_redeclaration";
  case ASTNode::Kind::Argument_ElementReplaceable:
    return "element_replaceable";
  case ASTNode::Kind::ArrayDimension:
    return "array_dimension";
  case ASTNode::Kind::Class_Function_PartialDerFunction:
    return "partial_der_function";
  case ASTNode::Kind::Class_Function_StandardFunction:
    return "standard_function";
  case ASTNode::Kind::Class_Model:
    return "model";
  case ASTNode::Kind::Class_Package:
    return "package";
  case ASTNode::Kind::Class_Record:
    return "record";
  case ASTNode::Kind::ClassModification:
    return "class_modification";
  case ASTNode::Kind::ComponentReferenceEntry:
    return "component_reference_entry";
  case ASTNode::Kind::Equation_Equality:
    return "equation_equality";
  case ASTNode::Kind::Equation_For:
    return "equation_for";
  case ASTNode::Kind::Equation_If:
    return "equation_if";
  case ASTNode::Kind::Equation_When:
    return "equation_when";
  case ASTNode::Kind::EquationSection:
    return "equation_section";
  case ASTNode::Kind::Expression_ArrayGenerator_ArrayConstant:
    return "array_constant";
  case ASTNode::Kind::Expression_ArrayGenerator_ArrayForGenerator:
    return "array_for_generator";
  case ASTNode::Kind::Expression_Call:
    return "call";
  case ASTNode::Kind::Expression_ComponentReference:
    return "component_reference";
  case ASTNode::Kind::Expression_Constant:
    return "constant";
  case ASTNode::Kind::Expression_Operation:
    return "operation";
  case ASTNode::Kind::Expression_Subscript:
    return "subscript";
  case ASTNode::Kind::Expression_Tuple:
    return "tuple";
  case ASTNode::Kind::ForIndex:
    return "for_index";
  case ASTNode::Kind::FunctionArgument_Expression:
    return "function_argument_expression";
  case ASTNode::Kind::FunctionArgument_Named:
    return "function_argument_named";
  case ASTNode::Kind::FunctionArgument_Reduction:
    return "function_argument_reduction";
  case ASTNode::Kind::Member:
    return "member";
  case ASTNode::Kind::Modification:
    return "modification";
  case ASTNode::Kind::Statement_Assignment:
    return "assignment_statement";
  case ASTNode::Kind::Statement_Break:
    return "break_statement";
  case ASTNode::Kind::Statement_For:
    return "for_statement";
  case ASTNode::Kind::Statement_If:
    return "if_statement";
  case ASTNode::Kind::Statement_Return:
    return "return_statement";
  case ASTNode::Kind::Statement_When:
    return "when_statement";
  case ASTNode::Kind::Statement_While:
    return "while_statement";
  case ASTNode::Kind::StatementsBlock:
    return "statements_block";
  case ASTNode::Kind::TypePrefix:
    return "type_prefix";
  case ASTNode::Kind::VariableType_BuiltIn:
    return "builtin_type";
  case ASTNode::Kind::VariableType_UserDefined:
    return "user_defined_type";
  default:
    llvm_unreachable("Unknown node kind");
    return "unknown";
  }
}
} // namespace

namespace marco::ast::bmodelica {
ASTNode::ASTNode(Kind kind, SourceRange location, ASTNode *parent)
    : kind(kind), location(std::move(location)), parent(parent) {}

ASTNode::ASTNode(const ASTNode &other) = default;

ASTNode::~ASTNode() = default;

const SourceRange &ASTNode::getLocation() const { return location; }

void ASTNode::setLocation(SourceRange loc) { this->location = std::move(loc); }

ASTNode *ASTNode::getParent() { return parent; }

const ASTNode *ASTNode::getParent() const { return parent; }

void ASTNode::setParent(ASTNode *node) { parent = node; }

void ASTNode::addJSONProperties(llvm::json::Object &obj) const {
  obj["node_type"] = toString(kind);
}
} // namespace marco::ast::bmodelica
