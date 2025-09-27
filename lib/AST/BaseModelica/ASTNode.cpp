#include "marco/AST/BaseModelica/ASTNode.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace {
static std::string toString(ASTNodeKind kind) {
  switch (kind) {
  case ASTNodeKind::Root:
    return "root";
  case ASTNodeKind::Algorithm:
    return "algorithm";
  case ASTNodeKind::Annotation:
    return "annotation";
  case ASTNodeKind::Argument_ElementModification:
    return "element_modification";
  case ASTNodeKind::Argument_ElementRedeclaration:
    return "element_redeclaration";
  case ASTNodeKind::Argument_ElementReplaceable:
    return "element_replaceable";
  case ASTNodeKind::ArrayDimension:
    return "array_dimension";
  case ASTNodeKind::Class_Function_PartialDerFunction:
    return "partial_der_function";
  case ASTNodeKind::Class_Function_StandardFunction:
    return "standard_function";
  case ASTNodeKind::Class_Model:
    return "model";
  case ASTNodeKind::Class_Package:
    return "package";
  case ASTNodeKind::Class_Record:
    return "record";
  case ASTNodeKind::ClassModification:
    return "class_modification";
  case ASTNodeKind::ComponentReferenceEntry:
    return "component_reference_entry";
  case ASTNodeKind::Equation_Equality:
    return "equation_equality";
  case ASTNodeKind::Equation_For:
    return "equation_for";
  case ASTNodeKind::Equation_If:
    return "equation_if";
  case ASTNodeKind::Equation_When:
    return "equation_when";
  case ASTNodeKind::EquationSection:
    return "equation_section";
  case ASTNodeKind::Expression_ArrayGenerator_ArrayConstant:
    return "array_constant";
  case ASTNodeKind::Expression_ArrayGenerator_ArrayForGenerator:
    return "array_for_generator";
  case ASTNodeKind::Expression_Call:
    return "call";
  case ASTNodeKind::Expression_ComponentReference:
    return "component_reference";
  case ASTNodeKind::Expression_Constant:
    return "constant";
  case ASTNodeKind::Expression_Operation:
    return "operation";
  case ASTNodeKind::Expression_Subscript:
    return "subscript";
  case ASTNodeKind::Expression_Tuple:
    return "tuple";
  case ASTNodeKind::ForIndex:
    return "for_index";
  case ASTNodeKind::FunctionArgument_Expression:
    return "function_argument_expression";
  case ASTNodeKind::FunctionArgument_Named:
    return "function_argument_named";
  case ASTNodeKind::FunctionArgument_Reduction:
    return "function_argument_reduction";
  case ASTNodeKind::Member:
    return "member";
  case ASTNodeKind::Modification:
    return "modification";
  case ASTNodeKind::Statement_Assignment:
    return "assignment_statement";
  case ASTNodeKind::Statement_Break:
    return "break_statement";
  case ASTNodeKind::Statement_For:
    return "for_statement";
  case ASTNodeKind::Statement_If:
    return "if_statement";
  case ASTNodeKind::Statement_Return:
    return "return_statement";
  case ASTNodeKind::Statement_When:
    return "when_statement";
  case ASTNodeKind::Statement_While:
    return "while_statement";
  case ASTNodeKind::StatementsBlock:
    return "statements_block";
  case ASTNodeKind::TypePrefix:
    return "type_prefix";
  case ASTNodeKind::VariableType_BuiltIn:
    return "builtin_type";
  case ASTNodeKind::VariableType_UserDefined:
    return "user_defined_type";
  default:
    llvm_unreachable("Unknown node kind");
    return "unknown";
  }
}
} // namespace

namespace marco::ast::bmodelica {
void addNodeKindToJSON(const ASTNode &node, llvm::json::Object &obj) {
  obj["kind"] = toString(node.getKind<ASTNodeKind>());
}
} // namespace marco::ast::bmodelica
