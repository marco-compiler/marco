#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ClassDependencyGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
Lowerer::Lowerer(BridgeInterface *bridge) : bridge(bridge) {}

Lowerer::~Lowerer() = default;

mlir::Location Lowerer::loc(const SourcePosition &location) {
  return mlir::FileLineColLoc::get(
      builder().getStringAttr(location.file->getFileName()), location.line,
      location.column);
}

mlir::Location Lowerer::loc(const SourceRange &location) {
  return loc(location.begin);
}

mlir::OpBuilder &Lowerer::builder() { return getContext().getBuilder(); }

mlir::SymbolTableCollection &Lowerer::getSymbolTable() {
  return getContext().getSymbolTable();
}

void Lowerer::getVisibleSymbols(
    mlir::Operation *scope, std::set<std::string> &visibleSymbols,
    llvm::function_ref<bool(mlir::Operation *)> filterFn) {
  // Check the global symbol table.
  mlir::SymbolTable::walkSymbolTables(
      getRoot(), true,
      [this, &scope, &visibleSymbols, &filterFn](mlir::Operation *op,
                                                 bool visible) {
        const mlir::StringAttr attr = op->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName());
        if (attr) {
          llvm::StringRef symbolName = attr;
          if (resolveSymbolName(symbolName, scope, filterFn)) {
            visibleSymbols.insert(symbolName.str());
          }
        }
      });

  // Check the variables symbol table.
  std::set<std::string> declaredVariables =
      getVariablesSymbolTable().getVariables(false);
  for (auto pVar = declaredVariables.cbegin(); pVar != declaredVariables.cend();
       ++pVar) {
    if (resolveSymbolName(llvm::StringRef(*pVar), scope, filterFn)) {
      visibleSymbols.insert(*pVar);
    }
  }
}

void Lowerer::getVisibleSymbols(mlir::Operation *scope,
                                std::set<std::string> &visibleSymbols) {
  return getVisibleSymbols(scope, visibleSymbols,
                           [](mlir::Operation *op) { return true; });
}

VariablesSymbolTable &Lowerer::getVariablesSymbolTable() {
  return getContext().getVariablesSymbolTable();
}

mlir::Operation *Lowerer::getLookupScope() {
  return getContext().getLookupScope();
}

void Lowerer::pushLookupScope(mlir::Operation *lookupScope) {
  getContext().pushLookupScope(lookupScope);
}

mlir::Operation *Lowerer::getClass(const ast::bmodelica::Class &cls) {
  llvm::SmallVector<const ast::bmodelica::Class *> classes;
  const ast::ASTNode *current = &cls;

  while (current != nullptr && !current->isa<ast::bmodelica::Root>()) {
    classes.push_back(current->cast<ast::bmodelica::Class>());
    current = current->getParentOfType<ast::bmodelica::Class>();
  }

  mlir::Operation *result = getRoot();

  while (!classes.empty() && result != nullptr) {
    const ast::bmodelica::Class *node = classes.back();

    result = getSymbolTable().lookupSymbolIn(
        result, builder().getStringAttr(node->getName()));

    classes.pop_back();
  }

  assert(result != nullptr && "Class not found");
  return result;
}

mlir::SymbolRefAttr Lowerer::getSymbolRefFromRoot(mlir::Operation *symbol) {
  llvm::SmallVector<mlir::FlatSymbolRefAttr> flatSymbolAttrs;

  flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
      builder().getContext(),
      mlir::cast<mlir::SymbolOpInterface>(symbol).getName()));

  mlir::Operation *parent = symbol->getParentOp();

  while (parent != nullptr) {
    if (auto classInterface = mlir::dyn_cast<ClassInterface>(parent)) {
      flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
          builder().getContext(),
          mlir::cast<mlir::SymbolOpInterface>(classInterface.getOperation())
              .getName()));
    }

    parent = parent->getParentOp();
  }

  std::reverse(flatSymbolAttrs.begin(), flatSymbolAttrs.end());

  return mlir::SymbolRefAttr::get(builder().getContext(),
                                  flatSymbolAttrs[0].getValue(),
                                  llvm::ArrayRef(flatSymbolAttrs).drop_front());
}

mlir::Operation *Lowerer::resolveClassName(llvm::StringRef name,
                                           mlir::Operation *currentScope) {
  return resolveSymbolName(name, currentScope, [](mlir::Operation *op) {
    return mlir::isa<ClassInterface>(op);
  });
}

std::optional<mlir::Operation *>
Lowerer::resolveType(const ast::bmodelica::UserDefinedType &type,
                     mlir::Operation *lookupScope) {
  mlir::Operation *scope = lookupScope;

  if (type.isGlobalLookup()) {
    scope = getRoot();
  }

  mlir::Operation *originalScope = scope;

  scope = resolveSymbolName<ClassInterface>(type.getElement(0), scope);

  if (!scope) {
    std::set<std::string> visibleTypes;
    getVisibleSymbols<ClassInterface>(originalScope, visibleTypes);

    emitIdentifierError(IdentifierError::IdentifierType::TYPE,
                        type.getElement(0), visibleTypes, type.getLocation());
    return std::nullopt;
  }

  for (size_t i = 1, e = type.getPathLength(); i < e && scope != nullptr; ++i) {
    scope = getSymbolTable().lookupSymbolIn(
        scope, builder().getStringAttr(type.getElement(i)));
  }

  return scope;
}

mlir::Operation *Lowerer::resolveTypeFromRoot(mlir::SymbolRefAttr name) {
  mlir::Operation *scope = getRoot();
  scope = getSymbolTable().lookupSymbolIn(scope, name.getRootReference());

  for (mlir::FlatSymbolRefAttr nestedRef : name.getNestedReferences()) {
    if (scope == nullptr) {
      return nullptr;
    }

    scope = getSymbolTable().lookupSymbolIn(scope, nestedRef.getAttr());
  }

  return scope;
}

mlir::Operation *Lowerer::resolveSymbolName(llvm::StringRef name,
                                            mlir::Operation *currentScope) {
  return resolveSymbolName(name, currentScope,
                           [](mlir::Operation *op) { return true; });
}

mlir::Operation *Lowerer::resolveSymbolName(
    llvm::StringRef name, mlir::Operation *currentScope,
    llvm::function_ref<bool(mlir::Operation *)> filterFn) {
  mlir::Operation *scope = currentScope;

  while (scope != nullptr) {
    if (scope->hasTrait<mlir::OpTrait::SymbolTable>()) {
      mlir::Operation *result =
          getSymbolTable().lookupSymbolIn(scope, builder().getStringAttr(name));

      if (result != nullptr && filterFn(result)) {
        return result;
      }
    }

    scope = scope->getParentWithTrait<mlir::OpTrait::SymbolTable>();
  }

  return nullptr;
}

std::optional<Reference> Lowerer::lookupVariable(llvm::StringRef name) {
  return getVariablesSymbolTable().lookup(name);
}

void Lowerer::insertVariable(llvm::StringRef name, Reference reference) {
  getVariablesSymbolTable().insert(name, reference);
}

bool Lowerer::isScalarType(mlir::Type type) {
  return mlir::isa<BooleanType, IntegerType, RealType, mlir::IndexType>(type);
}

LoweringContext &Lowerer::getContext() { return bridge->getContext(); }

const LoweringContext &Lowerer::getContext() const {
  return bridge->getContext();
}

mlir::Operation *Lowerer::getRoot() const { return bridge->getRoot(); }

void Lowerer::declare(const ast::bmodelica::Class &node) {
  return bridge->declare(node);
}

void Lowerer::declare(const ast::bmodelica::Model &node) {
  return bridge->declare(node);
}

void Lowerer::declare(const ast::bmodelica::Package &node) {
  return bridge->declare(node);
}

void Lowerer::declare(const ast::bmodelica::PartialDerFunction &node) {
  return bridge->declare(node);
}

void Lowerer::declare(const ast::bmodelica::Record &node) {
  return bridge->declare(node);
}

void Lowerer::declare(const ast::bmodelica::StandardFunction &node) {
  return bridge->declare(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::Class &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::Model &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::Package &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::PartialDerFunction &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::Record &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declareVariables(const ast::bmodelica::StandardFunction &node) {
  return bridge->declareVariables(node);
}

bool Lowerer::declare(const ast::bmodelica::Member &node) {
  return bridge->declare(node);
}

bool Lowerer::lower(const ast::bmodelica::Class &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::Model &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::Package &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::PartialDerFunction &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::Record &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::StandardFunction &node) {
  return bridge->lower(node);
}

bool Lowerer::lowerClassBody(const ast::bmodelica::Class &node) {
  return bridge->lowerClassBody(node);
}

bool Lowerer::createBindingEquation(
    const ast::bmodelica::Member &variable,
    const ast::bmodelica::Expression &expression) {
  return bridge->createBindingEquation(variable, expression);
}

bool Lowerer::lowerStartAttribute(mlir::SymbolRefAttr variable,
                                  const ast::bmodelica::Expression &expression,
                                  bool fixed, bool each) {
  return bridge->lowerStartAttribute(variable, expression, fixed, each);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::Expression &expression) {
  return bridge->lower(expression);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::ArrayGenerator &array) {
  return bridge->lower(array);
}

std::optional<Results> Lowerer::lower(const ast::bmodelica::Call &call) {
  return bridge->lower(call);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::Constant &constant) {
  return bridge->lower(constant);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::Operation &operation) {
  return bridge->lower(operation);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::ComponentReference &componentReference) {
  return bridge->lower(componentReference);
}

std::optional<Results> Lowerer::lower(const ast::bmodelica::Tuple &tuple) {
  return bridge->lower(tuple);
}

std::optional<Results>
Lowerer::lower(const ast::bmodelica::Subscript &subscript) {
  return bridge->lower(subscript);
}

bool Lowerer::lower(const ast::bmodelica::EquationSection &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::Equation &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::EqualityEquation &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::ForEquation &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::IfEquation &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::WhenEquation &node) {
  return bridge->lower(node);
}

bool Lowerer::lower(const ast::bmodelica::Algorithm &algorithm) {
  return bridge->lower(algorithm);
}

bool Lowerer::lower(const ast::bmodelica::Statement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::AssignmentStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lowerAssignmentToComponentReference(
    mlir::Location assignmentLoc,
    const ast::bmodelica::ComponentReference &destination, mlir::Value value) {
  return bridge->lowerAssignmentToComponentReference(assignmentLoc, destination,
                                                     value);
}

bool Lowerer::lower(const ast::bmodelica::BreakStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::CallStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::ForStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::IfStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::ReturnStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::WhenStatement &statement) {
  return bridge->lower(statement);
}

bool Lowerer::lower(const ast::bmodelica::WhileStatement &statement) {
  return bridge->lower(statement);
}

void Lowerer::emitIdentifierError(
    IdentifierError::IdentifierType identifierType, llvm::StringRef name,
    const std::set<std::string> &declaredIdentifiers,
    const marco::SourceRange &location) {
  IdentifierError error(identifierType, name, declaredIdentifiers);
  std::string actual = error.getActual();
  std::string predicted = error.getPredicted();

  std::string errorString = "Unknown ";
  switch (identifierType) {
  case marco::codegen::lowering::bmodelica::IdentifierError::IdentifierType::
      FUNCTION: {
    errorString += "function";
    break;
  }
  case marco::codegen::lowering::bmodelica::IdentifierError::IdentifierType::
      VARIABLE: {
    errorString += "variable";
    break;
  }
  case marco::codegen::lowering::bmodelica::IdentifierError::IdentifierType::
      TYPE: {
    errorString += "type or class";
    break;
  }
  case marco::codegen::lowering::bmodelica::IdentifierError::IdentifierType::
      FIELD: {
    errorString += "field";
    break;
  }
  default: {
    llvm_unreachable("Unkown error type.");
    break;
  }
  }
  errorString += " identifier " + actual + ".";

  if (predicted != "") {
    errorString += " Did you mean " + predicted + "?";
  }

  mlir::emitError(loc(location)) << errorString;
}
} // namespace marco::codegen::lowering::bmodelica
