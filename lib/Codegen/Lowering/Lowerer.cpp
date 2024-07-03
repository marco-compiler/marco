#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/ClassDependencyGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

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
        builder().getStringAttr(location.file->getFileName()),
        location.line,
        location.column);
  }

  mlir::Location Lowerer::loc(const SourceRange& location)
  {
    return loc(location.begin);
  }

  mlir::OpBuilder& Lowerer::builder()
  {
    return getContext().getBuilder();
  }

  mlir::SymbolTableCollection& Lowerer::getSymbolTable()
  {
    return getContext().getSymbolTable();
  }

  std::set<llvm::StringRef>& Lowerer::getDeclaredVariables()
  {
    return getContext().getDeclaredVariables();
  }

  void Lowerer::initializeDeclaredSymbols(
      mlir::Operation* scope, 
      std::set<std::string>& declaredSymbols,
      llvm::function_ref<bool(mlir::Operation*)> filterFn)
  {
    mlir::SymbolTable::walkSymbolTables(getRoot(), true, [this, &scope, &declaredSymbols, &filterFn](mlir::Operation *op, bool visible) {
      const mlir::StringAttr attr = op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());
      if (attr) {
        const llvm::StringRef symbolName = attr;
        if (resolveSymbolName(symbolName, scope, filterFn)) {
          declaredSymbols.insert(std::string(symbolName));
        }
      }
    });

    std::set<llvm::StringRef>& declaredVariables = getDeclaredVariables();
    for (auto pVar = declaredVariables.cbegin(); pVar != declaredVariables.cend(); ++pVar) {
      if (resolveSymbolName(*pVar, scope, filterFn)) {
        declaredSymbols.insert(std::string(*pVar));
      }
    }
  }

  void Lowerer::initializeDeclaredSymbols(
      mlir::Operation* scope, 
      std::set<std::string>& declaredSymbols)
  {
    return initializeDeclaredSymbols(scope, declaredSymbols, [](mlir::Operation* op) {
      return true;
    });
  }

  void Lowerer::initializeDeclaredVars(std::set<std::string>& declaredVars)
  {
    std::set<llvm::StringRef>& declaredVariables = getDeclaredVariables();
    for (auto pVar = declaredVariables.cbegin(); pVar != declaredVariables.cend(); ++pVar) {
      if (lookupVariable(*pVar)) {
        declaredVars.insert(std::string(*pVar));
      }
    }
  }

  LoweringContext::VariablesSymbolTable& Lowerer::getVariablesSymbolTable()
  {
    return getContext().getVariablesSymbolTable();
  }

  mlir::Operation* Lowerer::getLookupScope()
  {
    return getContext().getLookupScope();
  }

  void Lowerer::pushLookupScope(mlir::Operation* lookupScope)
  {
    getContext().pushLookupScope(lookupScope);
  }

  mlir::Operation* Lowerer::getClass(const ast::Class& cls)
  {
    llvm::SmallVector<const ast::Class*> classes;
    const ast::ASTNode* current = &cls;

    while (current != nullptr && !current->isa<ast::Root>()) {
      classes.push_back(current->cast<ast::Class>());
      current = current->getParentOfType<ast::Class>();
    }

    mlir::Operation* result = getRoot();

    while (!classes.empty() && result != nullptr) {
      const ast::Class* node = classes.back();

      result = getSymbolTable().lookupSymbolIn(
          result, builder().getStringAttr(node->getName()));

      classes.pop_back();
    }

    return result;
  }

  mlir::SymbolRefAttr Lowerer::getSymbolRefFromRoot(mlir::Operation* symbol)
  {
    llvm::SmallVector<mlir::FlatSymbolRefAttr> flatSymbolAttrs;

    flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
        builder().getContext(),
        mlir::cast<mlir::SymbolOpInterface>(symbol).getName()));

    mlir::Operation* parent = symbol->getParentOp();

    while (parent != nullptr) {
      if (auto classInterface = mlir::dyn_cast<ClassInterface>(parent)) {
        flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
            builder().getContext(),
            mlir::cast<mlir::SymbolOpInterface>(
                classInterface.getOperation()).getName()));
      }

      parent = parent->getParentOp();
    }

    std::reverse(flatSymbolAttrs.begin(), flatSymbolAttrs.end());

    return mlir::SymbolRefAttr::get(
        builder().getContext(), flatSymbolAttrs[0].getValue(),
        llvm::ArrayRef(flatSymbolAttrs).drop_front());
  }

  mlir::Operation* Lowerer::resolveClassName(
      llvm::StringRef name,
      mlir::Operation* currentScope)
  {
    return resolveSymbolName(name, currentScope, [](mlir::Operation* op) {
      return mlir::isa<ClassInterface>(op);
    });
  }

  std::optional<mlir::Operation*> Lowerer::resolveType(
      const ast::UserDefinedType& type,
      mlir::Operation* lookupScope)
  {
    mlir::Operation* scope = lookupScope;

    if (type.isGlobalLookup()) {
      scope = getRoot();
    }

    mlir::Operation* originalScope = scope;

    scope = resolveSymbolName<ClassInterface>(
        type.getElement(0), scope);

    if (!scope) {
      std::set<std::string> declaredTypes;
      initializeDeclaredSymbols<ClassInterface>(originalScope, declaredTypes);

      const marco::SourceRange sourceRange = type.getLocation();
          emitIdentifierError(IdentifierError::IdentifierType::TYPE, std::string(type.getElement(0)), 
                              declaredTypes, sourceRange);
      return std::nullopt;
    }

    for (size_t i = 1, e = type.getPathLength();
         i < e && scope != nullptr; ++i) {
      scope = getSymbolTable().lookupSymbolIn(
          scope, builder().getStringAttr(type.getElement(i)));
    }

    return scope;
  }

  mlir::Operation* Lowerer::resolveTypeFromRoot(mlir::SymbolRefAttr name)
  {
    mlir::Operation* scope = getRoot();
    scope = getSymbolTable().lookupSymbolIn(scope, name.getRootReference());

    for (mlir::FlatSymbolRefAttr nestedRef : name.getNestedReferences()) {
      if (scope == nullptr) {
        return nullptr;
      }

      scope = getSymbolTable().lookupSymbolIn(scope, nestedRef.getAttr());
    }

    return scope;
  }

  mlir::Operation* Lowerer::resolveSymbolName(
      llvm::StringRef name,
      mlir::Operation* currentScope)
  {
    return resolveSymbolName(name, currentScope, [](mlir::Operation* op) {
      return true;
    });
  }

  mlir::Operation* Lowerer::resolveSymbolName(
      llvm::StringRef name,
      mlir::Operation* currentScope,
      llvm::function_ref<bool(mlir::Operation*)> filterFn)
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

  std::optional<Reference> Lowerer::lookupVariable(llvm::StringRef name)
  {
    // Check if the variable is present in the symbol table.
    // If it is, return a Reference to it, otherwise return a null optional.
    const auto &symbolTable = getVariablesSymbolTable();
    if (symbolTable.count(name)) {
      return symbolTable.lookup(name);
    }
    return std::nullopt;
  }

  void Lowerer::insertVariable(llvm::StringRef name, Reference reference)
  {
    getDeclaredVariables().insert(name);
    getVariablesSymbolTable().insert(name, reference);
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

  bool Lowerer::declareVariables(const ast::Class& node)
  {
    return bridge->declareVariables(node);
  }

  bool Lowerer::declareVariables(const ast::Model& node)
  {
    return bridge->declareVariables(node);
  }

  bool Lowerer::declareVariables(const ast::Package& node)
  {
    return bridge->declareVariables(node);
  }

  void Lowerer::declareVariables(const ast::PartialDerFunction& node)
  {
    return bridge->declareVariables(node);
  }

  bool Lowerer::declareVariables(const ast::Record& node)
  {
    return bridge->declareVariables(node);
  }

  bool Lowerer::declareVariables(const ast::StandardFunction& node)
  {
    return bridge->declareVariables(node);
  }

  bool Lowerer::declare(const ast::Member& node)
  {
    return bridge->declare(node);
  }

  bool Lowerer::lower(const ast::Class& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::Model& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::Package& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::PartialDerFunction& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::Record& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::StandardFunction& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lowerClassBody(const ast::Class& node)
  {
    return bridge->lowerClassBody(node);
  }

  bool Lowerer::createBindingEquation(
      const ast::Member& variable,
      const ast::Expression& expression)
  {
    return bridge->createBindingEquation(variable, expression);
  }

  bool Lowerer::lowerStartAttribute(
      mlir::SymbolRefAttr variable,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    return bridge->lowerStartAttribute(variable, expression, fixed, each);
  }

  std::optional<Results> Lowerer::lower(const ast::Expression& expression)
  {
    return bridge->lower(expression);
  }

  std::optional<Results> Lowerer::lower(const ast::ArrayGenerator& array)
  {
    return bridge->lower(array);
  }

  std::optional<Results> Lowerer::lower(const ast::Call& call)
  {
    return bridge->lower(call);
  }

  Results Lowerer::lower(const ast::Constant& constant)
  {
    return bridge->lower(constant);
  }

  std::optional<Results> Lowerer::lower(const ast::Operation& operation)
  {
    return bridge->lower(operation);
  }

  std::optional<Results> Lowerer::lower(const ast::ComponentReference& componentReference)
  {
    return bridge->lower(componentReference);
  }

  std::optional<Results> Lowerer::lower(const ast::Tuple& tuple)
  {
    return bridge->lower(tuple);
  }

  std::optional<Results> Lowerer::lower(const ast::Subscript& subscript)
  {
    return bridge->lower(subscript);
  }

  bool Lowerer::lower(const ast::EquationSection& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::Equation& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::EqualityEquation& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::ForEquation& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::IfEquation& node)
  {
    return bridge->lower(node);
  }

  void Lowerer::lower(const ast::WhenEquation& node)
  {
    return bridge->lower(node);
  }

  bool Lowerer::lower(const ast::Algorithm& algorithm)
  {
    return bridge->lower(algorithm);
  }

  bool Lowerer::lower(const ast::Statement& statement)
  {
    return bridge->lower(statement);
  }

  bool Lowerer::lower(const ast::AssignmentStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::lower(const ast::BreakStatement& statement)
  {
    return bridge->lower(statement);
  }

  bool Lowerer::lower(const ast::ForStatement& statement)
  {
    return bridge->lower(statement);
  }

  bool Lowerer::lower(const ast::IfStatement& statement)
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

  bool Lowerer::lower(const ast::WhileStatement& statement)
  {
    return bridge->lower(statement);
  }

  void Lowerer::emitIdentifierError(IdentifierError::IdentifierType identifierType, std::string name, 
                                    const std::set<std::string> &declaredIdentifiers,
                                    const marco::SourceRange& location)
  {
    const IdentifierError error(identifierType, name, declaredIdentifiers);
    const std::string actual = error.getActual();
    const std::string predicted = error.getPredicted();

    std::string errorString = "Error in AST to MLIR conversion. Unknown ";
    switch (identifierType) {
      case marco::codegen::lowering::IdentifierError::IdentifierType::FUNCTION: {
        errorString += "function";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::VARIABLE: {
        errorString += "variable";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::TYPE: {
        errorString += "type or class";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::FIELD: {
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
    
    mlir::DiagnosticEngine& diag = getContext().getDiagEngine();
    diag.emit(loc(location), mlir::DiagnosticSeverity::Error) << errorString;
  }
}
