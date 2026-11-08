#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANCETREE_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANCETREE_H

#include "marco/Dialect/Modelica/IR/Ops.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <map>
#include <set>
#include <optional>
#include <string>

namespace mlir::modelica::instantiation {
class Component;
class Scope;

/// Value given by a modification, together with the context needed to
/// evaluate it.
struct ModifierValue {
  /// The class-level operation holding the value, or the 'break' marker.
  mutable BindOp op;

  /// The scope in which the modification is written.
  Scope *scope = nullptr;

  /// Index positions selecting the element of the value applying to an
  /// element of an array of instances, outermost first.
  llvm::SmallVector<unsigned> splitPositions;
};

/// Redeclaration of an element.
struct Redeclaration {
  /// The class, enumeration or component declaration.
  mlir::Operation *op = nullptr;

  /// The scope in which the redeclaration is written.
  Scope *scope = nullptr;
};

/// Modification of an element, possibly obtained by merging multiple ones.
struct Modifier {
  std::optional<ModifierValue> value;
  bool isEach = false;
  std::map<std::string, Modifier> children;
  std::map<std::string, Redeclaration> redeclarations;

  bool empty() const {
    return !value && children.empty() && redeclarations.empty();
  }

  /// Get the value, unless removed by 'break'.
  const ModifierValue *getValue() const;

  /// Get the modification of a child element, if any.
  const Modifier *getChild(llvm::StringRef name) const;

  /// Merge two modifiers, the outer one overriding the inner one.
  static Modifier merge(const Modifier &outer, const Modifier &inner);

  /// Apply the indices of an array of instances to the values that are not
  /// marked as 'each'.
  void split(llvm::ArrayRef<unsigned> positions);

  /// Build the modifier described by a class-level modification region,
  /// written in the given scope.
  static Modifier parse(mlir::Region &region, Scope *scope);
};

/// Get the textual representation of a class-level component reference,
/// used to compare connections syntactically.
std::string getReferenceText(mlir::Value value);

/// Predefined classes.
enum class PredefinedClass {
  None,
  Real,
  Integer,
  Boolean,
  String,
  StateSelect,
  AssertionLevel,
  ExternalObject,
  Connections,
  Clock
};

/// Element of a class: a class definition or a component declaration.
struct Element {
  enum class Kind { Class, Component };

  Kind kind;

  /// The declaration: a 'modelica.class', 'modelica.enumeration' or
  /// 'modelica.component' operation. Null for predefined classes.
  mlir::Operation *op = nullptr;

  /// The predefined class, for predefined classes.
  PredefinedClass predefined = PredefinedClass::None;

  /// The scope in which the declaration is written, where the names it uses
  /// are looked up.
  Scope *declScope = nullptr;

  /// The scope the element belongs to.
  Scope *owner = nullptr;

  /// The modification applied by the owner to the element.
  Modifier modifier;

  /// For class-extends redeclarations, the replaced element.
  Element *replaced = nullptr;

  /// For components, whether the element is the implicit inner element of
  /// an outer element without a matching inner one.
  bool isImplicitInner = false;

  /// Lazily computed data.
  Component *component = nullptr;
  Scope *classScope = nullptr;

  bool isClass() const { return kind == Kind::Class; }
  bool isComponent() const { return kind == Kind::Component; }

  llvm::StringRef getName() const;
};

/// A class, possibly modified, acting as scope for the lookup of names.
/// Classes instantiated as components refer to the
/// component, so that the references to their elements can be turned into
/// component references.
class Scope {
public:
  /// The definition: a 'modelica.class', a 'modelica.enumeration', or the
  /// module for the root of the class tree. Null for predefined classes.
  mlir::Operation *op = nullptr;

  PredefinedClass predefined = PredefinedClass::None;

  /// The enclosing scope.
  Scope *parent = nullptr;

  /// The modification applied to the class.
  Modifier modifier;

  /// The component whose class is this one, if any. Base classes share the
  /// component of the derived class.
  Component *instance = nullptr;

  /// For class-extends redeclarations, the replaced element.
  Element *replaced = nullptr;

  /// Dotted path of the class, used for naming functions and diagnostics.
  std::string path;

  bool isRoot() const;
  bool isEncapsulated() const;
  ClassKind getKind() const;
  mlir::Location getLoc() const;

  /// Primitive class at the base of the class, if the class is a simple
  /// type. The scope of such class holds the merged
  /// modification of the attributes.
  Scope *primitiveBase = nullptr;

  bool isPrimitive() const { return primitiveBase != nullptr; }

  /// Short class definitions with array dimensions or prefixes,
  /// outermost first.
  llvm::SmallVector<std::pair<ExtendsOp, Scope *>> shortClassExtends;

  /// Elements, including the inherited ones, in declaration order.
  llvm::SmallVector<Element *> elements;
  llvm::StringMap<Element *> elementsByName;

  /// Local elements only.
  llvm::StringMap<Element *> localElements;

  llvm::SmallVector<ImportOp> imports;
  llvm::SmallVector<Scope *> bases;

  /// Equation and algorithm sections and external clauses, including the
  /// inherited ones, with the scopes in which they are written.
  llvm::SmallVector<std::pair<mlir::Operation *, Scope *>> sections;

  /// Names of the components removed by selective model extension.
  llvm::StringSet<> deselected;

  /// Connections removed by selective model extension, identified by the
  /// textual representation of their arguments.
  std::set<std::pair<std::string, std::string>> deselectedConnections;

  /// For base classes, the class extending them.
  Scope *derived = nullptr;

  /// Literals of enumerations.
  llvm::SmallVector<std::string> literals;

  enum class State { Created, LocalsCreated, Expanding, Expanded, Failed };
  State state = State::Created;
};

/// Component of the instance tree.
class Component {
public:
  /// The declaration. Null for the root of a model or of a function.
  Element *element = nullptr;

  /// The enclosing instance. Null for the roots and for the components of
  /// packages, which are only evaluated.
  Component *parent = nullptr;

  std::string name;

  /// The merged modification of the component.
  Modifier modifier;

  /// The class of the component, instantiated with the modification.
  Scope *classScope = nullptr;

  /// Number of indices of the enclosing arrays of instances.
  unsigned indexOffset = 0;

  /// Number of declared dimensions.
  unsigned numDeclaredDims = 0;

  /// The dimensions, once evaluated. Dynamic dimensions are only allowed
  /// within functions.
  std::optional<llvm::SmallVector<int64_t>> dims;

  std::optional<bool> enabled;

  /// For outer components, the corresponding inner one.
  Component *inner = nullptr;

  std::optional<Variability> variability;
  std::optional<Causality> causality;
  bool isProtected = false;

  /// Implicit inner components added to the root.
  llvm::SmallVector<Element *> implicitInners;

  bool isRoot() const { return element == nullptr; }
  bool isPrimitive() const { return classScope && classScope->isPrimitive(); }
  bool isOuter() const;
  bool isInner() const;

  /// Number of indices of the enclosing arrays of instances and of the
  /// component itself, if it is an array of instances.
  unsigned getNumIndices() const {
    return indexOffset + (isPrimitive() ? 0 : numDeclaredDims);
  }

  /// Get the dotted path from the root, without the name of the root.
  std::string getPath() const;

  ComponentOp getDeclaration() const;
};
} // namespace mlir::modelica::instantiation

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANCETREE_H
