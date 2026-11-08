#ifndef MARCO_AST_MODELICA_CLASS_H
#define MARCO_AST_MODELICA_CLASS_H

#include "marco/AST/Modelica/Equation.h"

namespace marco::ast::modelica {
//===---------------------------------------------------------------------===//
// Prefixes
//===---------------------------------------------------------------------===//

/// Prefixes of an element declaration.
struct ElementPrefixes {
  bool isProtected = false;
  bool isRedeclare = false;
  bool isFinal = false;
  bool isInner = false;
  bool isOuter = false;
  bool isReplaceable = false;

  void addToJSON(llvm::json::Object &obj) const;
};

enum class ConnectorKind { None, Flow, Stream };
enum class Variability { None, Discrete, Parameter, Constant };
enum class Causality { None, Input, Output };

/// Type prefix of a component clause.
struct TypePrefix {
  ConnectorKind connectorKind = ConnectorKind::None;
  Variability variability = Variability::None;
  Causality causality = Causality::None;

  void addToJSON(llvm::json::Object &obj) const;
};

llvm::StringRef toString(ConnectorKind kind);
llvm::StringRef toString(Variability variability);
llvm::StringRef toString(Causality causality);

//===---------------------------------------------------------------------===//
// Elements
//===---------------------------------------------------------------------===//

/// Kind of class.
enum class ClassRestriction {
  Class,
  Model,
  Record,
  OperatorRecord,
  Block,
  Connector,
  ExpandableConnector,
  Type,
  Package,
  Function,
  Operator,
  OperatorFunction
};

llvm::StringRef toString(ClassRestriction restriction);

enum class Purity { Unspecified, Pure, Impure };

/// Literal of an enumeration type.
class EnumerationLiteral
    : public NodeImpl<EnumerationLiteral, Node, NodeKind::EnumerationLiteral> {
public:
  using NodeImpl::NodeImpl;

  llvm::StringRef getName() const { return name; }
  void setName(std::string newName) { name = std::move(newName); }

  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string name;
  Description description;
};

class ExternalClause;

/// Class definition, in one of its forms:
/// - long: `model A "description" <elements> end A;`
/// - class-extends: `model extends A(<modification>) <elements> end A;`
/// - short: `type A = input B[3](<modification>) "description";`
/// - enumeration: `type E = enumeration(a, b) "description";`
/// - derivative: `function df = der(f, x, y) "description";`
class ClassDefinition
    : public NodeImpl<ClassDefinition, Node, NodeKind::ClassDefinition> {
public:
  enum class Form { Long, ClassExtends, Short, Enumeration, Derivative };

  using NodeImpl::NodeImpl;

  ElementPrefixes &getPrefixes() { return prefixes; }
  const ElementPrefixes &getPrefixes() const { return prefixes; }

  bool isEncapsulated() const { return encapsulated; }
  void setEncapsulated(bool value) { encapsulated = value; }

  bool isPartial() const { return partial; }
  void setPartial(bool value) { partial = value; }

  Purity getPurity() const { return purity; }
  void setPurity(Purity value) { purity = value; }

  ClassRestriction getRestriction() const { return restriction; }
  void setRestriction(ClassRestriction value) { restriction = value; }

  llvm::StringRef getName() const { return name; }
  void setName(std::string newName) { name = std::move(newName); }

  Form getForm() const { return form; }
  void setForm(Form value) { form = value; }

  /// Description string and annotation. For long class definitions, the
  /// annotation is the one at the end of the composition.
  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

  /// @name Long and class-extends definitions
  /// {

  /// Get the elements and the sections, in declaration order. Elements are
  /// ClassDefinition, ComponentClause, ExtendsClause and ImportClause nodes,
  /// while sections are EquationSection and AlgorithmSection nodes.
  const Children<Node> &getElements() const { return elements; }
  void addElement(std::unique_ptr<Node> node) {
    elements.emplace_back(std::move(node));
  }

  ExternalClause *getExternalClause() const { return externalClause.get(); }
  void setExternalClause(std::unique_ptr<ExternalClause> node);

  /// }
  /// @name Class-extends and short definitions
  /// {

  /// Modification of the inherited class (class-extends) or of the base
  /// class (short definitions).
  ClassModification *getClassModification() const {
    return classModification.get();
  }

  void setClassModification(std::unique_ptr<ClassModification> node) {
    classModification = std::move(node);
  }

  /// }
  /// @name Short definitions
  /// {

  Causality getBasePrefix() const { return basePrefix; }
  void setBasePrefix(Causality value) { basePrefix = value; }

  const Name &getBaseClass() const { return baseClass; }
  void setBaseClass(Name name) { baseClass = std::move(name); }

  const Children<Expression> &getSubscripts() const { return subscripts; }
  void addSubscript(std::unique_ptr<Expression> node) {
    subscripts.emplace_back(std::move(node));
  }

  /// }
  /// @name Enumerations
  /// {

  const Children<EnumerationLiteral> &getEnumerationLiterals() const {
    return enumerationLiterals;
  }

  void addEnumerationLiteral(std::unique_ptr<EnumerationLiteral> node) {
    enumerationLiterals.emplace_back(std::move(node));
  }

  /// Whether the enumeration is `enumeration(:)`.
  bool isUnspecifiedEnumeration() const { return unspecifiedEnumeration; }
  void setUnspecifiedEnumeration(bool value) { unspecifiedEnumeration = value; }

  /// }
  /// @name Derivative definitions
  /// {

  const Name &getDerivedFunction() const { return derivedFunction; }
  void setDerivedFunction(Name name) { derivedFunction = std::move(name); }

  llvm::ArrayRef<std::string> getDerivativeArguments() const {
    return derivativeArguments;
  }

  void addDerivativeArgument(std::string arg) {
    derivativeArguments.push_back(std::move(arg));
  }

  /// }

  ConstrainingClause *getConstrainingClause() const {
    return constrainingClause.get();
  }

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node) {
    constrainingClause = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  ElementPrefixes prefixes;
  bool encapsulated = false;
  bool partial = false;
  Purity purity = Purity::Unspecified;
  ClassRestriction restriction = ClassRestriction::Class;
  std::string name;
  Form form = Form::Long;
  Description description;
  Children<Node> elements;
  Child<ExternalClause> externalClause;
  Child<ClassModification> classModification;
  Causality basePrefix = Causality::None;
  Name baseClass;
  Children<Expression> subscripts;
  Children<EnumerationLiteral> enumerationLiterals;
  bool unspecifiedEnumeration = false;
  Name derivedFunction;
  std::vector<std::string> derivativeArguments;
  Child<ConstrainingClause> constrainingClause;
};

/// Single declaration of a component clause (`x[2](start = 1) = 3 if c`).
class ComponentDeclaration
    : public NodeImpl<ComponentDeclaration, Node,
                      NodeKind::ComponentDeclaration> {
public:
  using NodeImpl::NodeImpl;

  llvm::StringRef getName() const { return name; }
  void setName(std::string newName) { name = std::move(newName); }

  const Children<Expression> &getSubscripts() const { return subscripts; }
  void addSubscript(std::unique_ptr<Expression> node) {
    subscripts.emplace_back(std::move(node));
  }

  Modification *getModification() const { return modification.get(); }
  void setModification(std::unique_ptr<Modification> node) {
    modification = std::move(node);
  }

  /// Get the condition of a conditional component.
  Expression *getCondition() const { return condition.get(); }
  void setCondition(std::unique_ptr<Expression> node) {
    condition = std::move(node);
  }

  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string name;
  Children<Expression> subscripts;
  Child<Modification> modification;
  Child<Expression> condition;
  Description description;
};

/// Component clause, declaring one or more components with
/// the same prefixes and type.
class ComponentClause
    : public NodeImpl<ComponentClause, Node, NodeKind::ComponentClause> {
public:
  using NodeImpl::NodeImpl;

  ElementPrefixes &getPrefixes() { return prefixes; }
  const ElementPrefixes &getPrefixes() const { return prefixes; }

  TypePrefix &getTypePrefix() { return typePrefix; }
  const TypePrefix &getTypePrefix() const { return typePrefix; }

  const Name &getTypeSpecifier() const { return typeSpecifier; }
  void setTypeSpecifier(Name name) { typeSpecifier = std::move(name); }

  /// Array subscripts of the type specifier, which apply to all the
  /// declarations (e.g., `Real[3] a, b;`).
  const Children<Expression> &getSubscripts() const { return subscripts; }
  void addSubscript(std::unique_ptr<Expression> node) {
    subscripts.emplace_back(std::move(node));
  }

  const Children<ComponentDeclaration> &getDeclarations() const {
    return declarations;
  }

  void addDeclaration(std::unique_ptr<ComponentDeclaration> node) {
    declarations.emplace_back(std::move(node));
  }

  ConstrainingClause *getConstrainingClause() const {
    return constrainingClause.get();
  }

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node) {
    constrainingClause = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  ElementPrefixes prefixes;
  TypePrefix typePrefix;
  Name typeSpecifier;
  Children<Expression> subscripts;
  Children<ComponentDeclaration> declarations;
  Child<ConstrainingClause> constrainingClause;
};

/// Extends-clause. The modification may contain
/// inheritance modifications.
class ExtendsClause
    : public NodeImpl<ExtendsClause, Node, NodeKind::ExtendsClause> {
public:
  using NodeImpl::NodeImpl;

  bool isProtected() const { return protectedElement; }
  void setProtected(bool value) { protectedElement = value; }

  const Name &getBaseClass() const { return baseClass; }
  void setBaseClass(Name name) { baseClass = std::move(name); }

  ClassModification *getClassModification() const {
    return classModification.get();
  }

  void setClassModification(std::unique_ptr<ClassModification> node) {
    classModification = std::move(node);
  }

  ClassModification *getAnnotation() const { return annotation.get(); }
  void setAnnotation(std::unique_ptr<ClassModification> node) {
    annotation = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool protectedElement = false;
  Name baseClass;
  Child<ClassModification> classModification;
  Child<ClassModification> annotation;
};

/// Import-clause.
class ImportClause
    : public NodeImpl<ImportClause, Node, NodeKind::ImportClause> {
public:
  enum class Form {
    /// `import A.B;`
    Qualified,
    /// `import X = A.B;`
    Renaming,
    /// `import A.B.*;`
    Unqualified,
    /// `import A.B.{C, D};`
    Multiple
  };

  using NodeImpl::NodeImpl;

  bool isProtected() const { return protectedElement; }
  void setProtected(bool value) { protectedElement = value; }

  Form getForm() const { return form; }
  void setForm(Form value) { form = value; }

  const Name &getName() const { return name; }
  void setName(Name newName) { name = std::move(newName); }

  llvm::StringRef getAlias() const { return alias; }
  void setAlias(std::string value) { alias = std::move(value); }

  llvm::ArrayRef<std::string> getImportedNames() const { return importedNames; }
  void addImportedName(std::string value) {
    importedNames.push_back(std::move(value));
  }

  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool protectedElement = false;
  Form form = Form::Qualified;
  Name name;
  std::string alias;
  std::vector<std::string> importedNames;
  Description description;
};

/// Equation section.
class EquationSection
    : public NodeImpl<EquationSection, Node, NodeKind::EquationSection> {
public:
  using NodeImpl::NodeImpl;

  bool isInitial() const { return initial; }
  void setInitial(bool value) { initial = value; }

  const Children<Equation> &getEquations() const { return equations; }
  void addEquation(std::unique_ptr<Equation> node) {
    equations.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool initial = false;
  Children<Equation> equations;
};

/// Algorithm section.
class AlgorithmSection
    : public NodeImpl<AlgorithmSection, Node, NodeKind::AlgorithmSection> {
public:
  using NodeImpl::NodeImpl;

  bool isInitial() const { return initial; }
  void setInitial(bool value) { initial = value; }

  const Children<Statement> &getStatements() const { return statements; }
  void addStatement(std::unique_ptr<Statement> node) {
    statements.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool initial = false;
  Children<Statement> statements;
};

/// External function clause.
class ExternalClause
    : public NodeImpl<ExternalClause, Node, NodeKind::ExternalClause> {
public:
  using NodeImpl::NodeImpl;

  /// Get the language specification, if any.
  const std::optional<std::string> &getLanguage() const { return language; }
  void setLanguage(std::optional<std::string> value) {
    language = std::move(value);
  }

  /// Whether the external function call is specified.
  bool hasFunctionCall() const { return !function.empty(); }

  llvm::StringRef getFunction() const { return function; }
  void setFunction(std::string name) { function = std::move(name); }

  /// Get the component reference receiving the result, if any.
  ComponentReference *getOutput() const { return output.get(); }
  void setOutput(std::unique_ptr<ComponentReference> node) {
    output = std::move(node);
  }

  const Children<Expression> &getArguments() const { return arguments; }
  void addArgument(std::unique_ptr<Expression> node) {
    arguments.emplace_back(std::move(node));
  }

  ClassModification *getAnnotation() const { return annotation.get(); }
  void setAnnotation(std::unique_ptr<ClassModification> node) {
    annotation = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::optional<std::string> language;
  std::string function;
  Child<ComponentReference> output;
  Children<Expression> arguments;
  Child<ClassModification> annotation;
};

//===---------------------------------------------------------------------===//
// Root
//===---------------------------------------------------------------------===//

/// Content of a file: an optional within-clause and the
/// class definitions.
class StoredDefinition
    : public NodeImpl<StoredDefinition, Node, NodeKind::StoredDefinition> {
public:
  using NodeImpl::NodeImpl;

  /// Get the package specified by the within-clause, if any. An empty name
  /// means the top-level (`within;`).
  const std::optional<Name> &getWithin() const { return within; }
  void setWithin(std::optional<Name> name) { within = std::move(name); }

  const Children<ClassDefinition> &getClasses() const { return classes; }
  void addClass(std::unique_ptr<ClassDefinition> node) {
    classes.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::optional<Name> within;
  Children<ClassDefinition> classes;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CLASS_H
