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

mlir::SymbolTableCollection &Lowerer::getSymbolTables() {
  return getContext().getSymbolTables();
}

ScopedSymbolTable &Lowerer::getScopedSymbolTable() {
  return getContext().getScopedSymbolTable();
}

const ScopedSymbolTable &Lowerer::getScopedSymbolTable() const {
  return getContext().getScopedSymbolTable();
}

mlir::Operation *Lowerer::getLookupScope() const {
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

    result = getSymbolTables().lookupSymbolIn(
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

  scope = resolveSymbolName<ClassInterface>(type.getElement(0), scope);

  if (!scope) {
    emitUndeclaredTypeError(type.getElement(0), loc(type.getLocation()));
    return std::nullopt;
  }

  for (size_t i = 1, e = type.getPathLength(); i < e && scope != nullptr; ++i) {
    scope = getSymbolTables().lookupSymbolIn(
        scope, builder().getStringAttr(type.getElement(i)));
  }

  return scope;
}

mlir::Operation *Lowerer::resolveTypeFromRoot(mlir::SymbolRefAttr name) {
  mlir::Operation *scope = getRoot();
  scope = getSymbolTables().lookupSymbolIn(scope, name.getRootReference());

  for (mlir::FlatSymbolRefAttr nestedRef : name.getNestedReferences()) {
    if (scope == nullptr) {
      return nullptr;
    }

    scope = getSymbolTables().lookupSymbolIn(scope, nestedRef.getAttr());
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
      mlir::Operation *result = getSymbolTables().lookupSymbolIn(
          scope, builder().getStringAttr(name));

      if (result != nullptr && filterFn(result)) {
        return result;
      }
    }

    scope = scope->getParentWithTrait<mlir::OpTrait::SymbolTable>();
  }

  return nullptr;
}

std::optional<Reference> Lowerer::lookupVariable(llvm::StringRef name) {
  auto symbolInfo = getScopedSymbolTable().lookup(name);

  if (!symbolInfo) {
    return std::nullopt;
  }

  return symbolInfo->reference;
}

void Lowerer::insertVariable(llvm::StringRef name, Reference reference) {
  getScopedSymbolTable().insert(name, reference, SymbolType::Variable);
}

void Lowerer::insertVariableBuiltIn(llvm::StringRef name, Reference reference) {
  getScopedSymbolTable().insert(name, reference, SymbolType::VariableBuiltIn);
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

void Lowerer::emitUndeclaredClassError(llvm::StringRef name,
                                       mlir::Location location) const {
  emitUndeclaredSymbolError(
      name, location,
      getClassNamesVisibleFromScope(getLookupScope(),
                                    [](mlir::Operation *op) { return true; }));
}

void Lowerer::emitUndeclaredFunctionError(llvm::StringRef name,
                                          mlir::Location location,
                                          int64_t numArguments) const {
  llvm::StringSet<> symbols = getFunctionNamesVisibleFromScope(
      getLookupScope(), [&](mlir::Operation *op) {
        // Include only the functions with the same number of arguments as the
        // undeclared function.
        auto cls = mlir::cast<ClassInterface>(op);

        return llvm::count_if(cls.getBody().getOps<VariableOp>(),
                              [&](VariableOp variableOp) {
                                return variableOp.isInput();
                              }) == numArguments;
      });

  for (const auto &symbol : symbols) {
    llvm::outs() << "Available symbol: " << symbol.first() << "\n";
  }

  // Add the built-in functions to the list of available symbols, according to
  // the number of arguments.
  if (numArguments == 1) {
    symbols.insert("abs");
    symbols.insert("acos");
    symbols.insert("asin");
    symbols.insert("atan");
    symbols.insert("atan2");
    symbols.insert("ceil");
    symbols.insert("cos");
    symbols.insert("cosh");

    if (mlir::isa<ModelOp>(getLookupScope())) {
      symbols.insert("der");
    }

    symbols.insert("diagonal");
    symbols.insert("exp");
    symbols.insert("floor");
    symbols.insert("identity");
    symbols.insert("integer");
    symbols.insert("log");
    symbols.insert("log10");
    symbols.insert("ndims");
    symbols.insert("product");
    symbols.insert("sign");
    symbols.insert("sin");
    symbols.insert("sinh");
    symbols.insert("sqrt");
    symbols.insert("sum");
    symbols.insert("symmetric");
    symbols.insert("tan");
    symbols.insert("tanh");
    symbols.insert("transpose");
  } else if (numArguments == 2) {
    symbols.insert("div");
    symbols.insert("mod");
    symbols.insert("rem");
  } else if (numArguments == 3) {
    symbols.insert("linspace");
  }

  if (numArguments >= 1) {
    symbols.insert("ones");
    symbols.insert("zeros");
  }

  if (numArguments >= 1 && numArguments <= 2) {
    symbols.insert("max");
    symbols.insert("min");
    symbols.insert("size");
  }

  if (numArguments >= 2) {
    symbols.insert("fill");
  }

  // Emit the error.
  emitUndeclaredSymbolError(name, location, symbols);
}

void Lowerer::emitUndeclaredTypeError(llvm::StringRef name,
                                      mlir::Location location) const {
  llvm::StringSet<> symbols =
      getClassNamesVisibleFromScope(getLookupScope(), [](mlir::Operation *op) {
        return mlir::isa<RecordOp>(op);
      });

  symbols.insert("Boolean");
  symbols.insert("Integer");
  symbols.insert("Real");
  symbols.insert("String");

  emitUndeclaredSymbolError(name, location, symbols);
}

void Lowerer::emitUndeclaredVariableError(llvm::StringRef name,
                                          mlir::Location location) const {
  bool timeVariable = mlir::isa<ModelOp>(getLookupScope());

  llvm::StringSet<> availableSymbols = getScopedSymbolTable().getSymbolNames(
      [&](llvm::StringRef symbolName, Reference reference, SymbolType type) {
        if (symbolName == "time" && type == SymbolType::VariableBuiltIn &&
            !timeVariable) {
          return false;
        }

        return type == SymbolType::Variable ||
               type == SymbolType::VariableBuiltIn;
      });

  emitUndeclaredSymbolError(name, location, availableSymbols, 2);
}

void Lowerer::emitUndeclaredComponentError(llvm::StringRef name,
                                           mlir::Location location,
                                           mlir::Operation *parent) const {
  llvm::StringSet<> commponentNames;
  auto classInterface = mlir::dyn_cast<ClassInterface>(parent);

  for (VariableOp variableOp : classInterface.getBody().getOps<VariableOp>()) {
    commponentNames.insert(variableOp.getName());
  }

  emitUndeclaredSymbolError(name, location, commponentNames);
}

void Lowerer::emitUndeclaredSymbolError(
    llvm::StringRef name, mlir::Location loc,
    const llvm::StringSet<> &availableSymbols, unsigned int maxDistance) const {
  std::string message = "'" + name.str() + "' was not declared in this scope";

  if (!availableSymbols.empty()) {
    llvm::StringRef closestSymbol;
    unsigned int minDistance = std::numeric_limits<unsigned int>::max();

    for (llvm::StringRef symbol : availableSymbols.keys()) {
      unsigned int distance = name.edit_distance_insensitive(symbol);

      if (distance <= maxDistance && distance < minDistance) {
        minDistance = distance;
        closestSymbol = symbol;
      }
    }

    if (!closestSymbol.empty()) {
      message += "; did you mean '" + closestSymbol.str() + "'?";
    }
  }

  mlir::emitError(loc, message);
}

llvm::StringSet<> Lowerer::getAllClassNames(
    llvm::function_ref<bool(mlir::Operation *)> filterFn) const {
  return getClassNamesWithRoot(getRoot(), filterFn,
                               [](mlir::Operation *op) { return true; });
}

llvm::StringSet<> Lowerer::getAllFunctionNames(
    llvm::function_ref<bool(mlir::Operation *)> filterFn) const {
  return getClassNamesWithRoot(
      getRoot(),
      [&](mlir::Operation *op) {
        return mlir::isa<FunctionOp, DerFunctionOp>(op) && filterFn(op);
      },
      [](mlir::Operation *op) { return true; });
}

llvm::StringSet<> Lowerer::getClassNamesWithRoot(
    mlir::Operation *root, llvm::function_ref<bool(mlir::Operation *)> filterFn,
    llvm::function_ref<bool(mlir::Operation *)> visitFn) const {
  llvm::StringSet<> result;
  llvm::SmallVector<std::pair<ClassInterface, std::string>> worklist;

  if (visitFn(root)) {
    if (mlir::isa<mlir::ModuleOp>(root)) {
      for (ClassInterface cls :
           mlir::cast<mlir::ModuleOp>(root).getOps<ClassInterface>()) {
        std::string name = cls.getClassName().str();
        worklist.emplace_back(cls, name);

        if (filterFn(cls)) {
          result.insert(name);
        }
      }
    } else if (auto cls = mlir::dyn_cast<ClassInterface>(root)) {
      std::string name = cls.getClassName().str();
      worklist.emplace_back(cls, name);

      if (filterFn(cls)) {
        result.insert(name);
      }
    }
  }

  while (!worklist.empty()) {
    auto [cls, prefix] = worklist.pop_back_val();

    for (ClassInterface innerCls : cls.getBody().getOps<ClassInterface>()) {
      if (visitFn(innerCls)) {
        std::string name = prefix + "." + cls.getClassName().str();
        worklist.emplace_back(innerCls, name);

        if (filterFn(cls)) {
          result.insert(name);
        }
      }
    }
  }

  return result;
}

llvm::StringSet<> Lowerer::getClassNamesVisibleFromScope(
    mlir::Operation *scope,
    llvm::function_ref<bool(mlir::Operation *)> filterFn) const {
  llvm::StringSet<> result;
  llvm::DenseSet<mlir::Operation *> visited;

  while (scope) {
    llvm::StringSet<> names =
        getClassNamesWithRoot(scope, filterFn, [&](mlir::Operation *op) {
          return !visited.contains(op);
        });

    visited.insert(scope);

    for (const auto &name : names) {
      result.insert(name.first());
    }

    scope = scope->getParentWithTrait<ClassInterface::Trait>();
  }

  for (const auto &name : getAllClassNames(filterFn)) {
    if (!result.contains(name.first())) {
      result.insert(name.first());
    } else {
      result.insert("." + name.first().str());
    }
  }

  return result;
}

llvm::StringSet<> Lowerer::getFunctionNamesVisibleFromScope(
    mlir::Operation *scope,
    llvm::function_ref<bool(mlir::Operation *)> filterFn) const {
  return getClassNamesVisibleFromScope(scope, [&](mlir::Operation *op) {
    return mlir::isa<FunctionOp, DerFunctionOp>(op) && filterFn(op);
  });
}
} // namespace marco::codegen::lowering::bmodelica
