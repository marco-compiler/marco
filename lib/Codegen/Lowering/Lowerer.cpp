#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/ClassDependencyGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include <stack>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  Lowerer::Lowerer(BridgeInterface* bridge)
      : bridge(bridge)
  {
  }

  Lowerer::~Lowerer() = default;

  mlir::Location Lowerer::loc(const SourcePosition& location)
  {
    return mlir::FileLineColLoc::get(
        builder().getStringAttr(location.file->filePath()),
        location.line,
        location.column);
  }

  mlir::Location Lowerer::loc(const SourceRange& location)
  {
    return loc(location.begin);
  }

  mlir::OpBuilder& Lowerer::builder()
  {
    return getContext().builder;
  }

  mlir::SymbolTableCollection& Lowerer::getSymbolTable()
  {
    return getContext().symbolTable;
  }

  LoweringContext::VariablesSymbolTable& Lowerer::getVariablesSymbolTable()
  {
    return getContext().variablesSymbolTable;
  }

  mlir::Operation* Lowerer::getClass(const ast::Class& cls)
  {
    std::stack<const ast::Class*> classes;
    const ast::ASTNode* current = &cls;

    while (current != nullptr && !current->isa<ast::Root>()) {
      classes.push(current->cast<ast::Class>());
      current = current->getParentOfType<ast::Class>();
    }

    mlir::Operation* result = getRoot();

    while (!classes.empty() && result != nullptr) {
      const ast::Class* node = classes.top();

      result = getSymbolTable().lookupSymbolIn(
          result, builder().getStringAttr(node->getName()));

      classes.pop();
    }

    assert(result != nullptr && "Class not found");
    return result;
  }

  mlir::Operation* Lowerer::resolveClassName(
      llvm::StringRef name,
      mlir::Operation* currentScope)
  {
    return resolveSymbolName(name, currentScope, [](mlir::Operation* op) {
      return mlir::isa<ClassInterface>(op);
    });
  }

  mlir::Operation* Lowerer::resolveSymbolName(
      llvm::StringRef name,
      mlir::Operation* currentScope,
      std::function<bool(mlir::Operation*)> filterFn)
  {
    mlir::Operation* scope = currentScope;

    while (scope != nullptr) {
      if (scope->hasTrait<mlir::OpTrait::SymbolTable>()) {
        mlir::Operation* result = getSymbolTable().lookupSymbolIn(
            scope, builder().getStringAttr(name));

        if (result != nullptr && filterFn(result)) {
          return result;
        }
      }

      scope = scope->getParentWithTrait<mlir::OpTrait::SymbolTable>();
    }

    return nullptr;
  }

  Reference Lowerer::lookupVariable(llvm::StringRef name)
  {
    return getVariablesSymbolTable().lookup(name);
  }

  void Lowerer::insertVariable(llvm::StringRef name, Reference reference)
  {
    getVariablesSymbolTable().insert(name, reference);
  }

  mlir::Type Lowerer::getMostGenericScalarType(mlir::Type first, mlir::Type second)
  {
    assert((first.isa<BooleanType, IntegerType, RealType, mlir::IndexType>()));

    assert((second.isa<
            BooleanType, IntegerType, RealType, mlir::IndexType>()));

    if (first.isa<BooleanType>()) {
      if (second.isa<mlir::IndexType>()) {
        return IntegerType::get(first.getContext());
      }

      return second;
    }

    if (first.isa<IntegerType>()) {
      if (second.isa<RealType>()) {
        return second;
      }

      return first;
    }

    if (first.isa<RealType>()) {
      return first;
    }

    assert(first.isa<mlir::IndexType>());

    if (second.isa<RealType>() || second.isa<mlir::IndexType>()) {
      return second;
    }

    return IntegerType::get(first.getContext());
  }

  bool Lowerer::isScalarType(mlir::Type type)
  {
    return type.isa<BooleanType, IntegerType, RealType, mlir::IndexType>();
  }

  LoweringContext& Lowerer::getContext()
  {
    return bridge->getContext();
  }

  const LoweringContext& Lowerer::getContext() const
  {
    return bridge->getContext();
  }

  mlir::Operation* Lowerer::getRoot() const
  {
    return bridge->getRoot();
  }

  void Lowerer::declare(const ast::Class& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declare(const ast::Model& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declare(const ast::Package& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declare(const ast::PartialDerFunction& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declare(const ast::Record& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declare(const ast::StandardFunction& node)
  {
    return bridge->declare(node);
  }

  void Lowerer::declareClassVariables(const ast::Class& node)
  {
    return bridge->declareClassVariables(node);
  }

  void Lowerer::lower(const ast::Class& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::Model& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::Package& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::PartialDerFunction& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::Record& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::StandardFunction& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lowerClassBody(const ast::Class& node)
  {
    return bridge->lowerClassBody(node);
  }

  void Lowerer::createBindingEquation(
      const ast::Member& variable,
      const ast::Expression& expression)
  {
    return bridge->createBindingEquation(variable, expression);
  }

  void Lowerer::lowerStartAttribute(
      const ast::Member& variable,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    return bridge->lowerStartAttribute(variable, expression, fixed, each);
  }

  Results Lowerer::lower(const ast::Expression& expression)
  {
    return bridge->lower(expression);
  }

  Results Lowerer::lower(const ast::Array& array)
  {
    return bridge->lower(array);
  }

  Results Lowerer::lower(const ast::Call& call)
  {
    return bridge->lower(call);
  }

  Results Lowerer::lower(const ast::Constant& constant)
  {
    return bridge->lower(constant);
  }

  Results Lowerer::lower(const ast::Operation& operation)
  {
    return bridge->lower(operation);
  }

  Results Lowerer::lower(const ast::ReferenceAccess& referenceAccess)
  {
    return bridge->lower(referenceAccess);
  }

  Results Lowerer::lower(const ast::Tuple& tuple)
  {
    return bridge->lower(tuple);
  }

  void Lowerer::lower(const ast::Algorithm& algorithm)
  {
    return bridge->lower(algorithm);
  }

  void Lowerer::lower(const ast::Statement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::AssignmentStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::BreakStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::ForStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::IfStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::ReturnStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::WhenStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::WhileStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::Equation& equation, bool initialEquation)
  {
    return bridge->lower(equation, initialEquation);
  }

  void Lowerer::lower(
      const ast::ForEquation& forEquation, bool initialEquation)
  {
    return bridge->lower(forEquation, initialEquation);
  }
}
