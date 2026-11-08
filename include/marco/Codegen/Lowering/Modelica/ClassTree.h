#ifndef MARCO_CODEGEN_LOWERING_MODELICA_CLASSTREE_H
#define MARCO_CODEGEN_LOWERING_MODELICA_CLASSTREE_H

#include "marco/AST/Modelica/AST.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <optional>
#include <vector>

namespace marco::codegen::lowering::modelica {
/// Classes predefined by the language.
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

class ClassNode;

/// Element found by a lookup: either a class or a component.
struct Element {
  enum class Kind { Class, Component };

  Kind kind = Kind::Class;

  /// The class, for elements of kind 'Class'.
  ClassNode *classNode = nullptr;

  /// The declaration, for elements of kind 'Component'.
  const ast::modelica::ComponentClause *clause = nullptr;
  const ast::modelica::ComponentDeclaration *declaration = nullptr;

  /// The class declaring the element.
  ClassNode *owner = nullptr;

  /// For inherited elements, the class whose lookup found the element. The
  /// names used by the declaration are resolved in this class first, so that
  /// the redeclarations of the derived class take effect.
  ClassNode *accessScope = nullptr;

  bool isClass() const { return kind == Kind::Class; }
  bool isComponent() const { return kind == Kind::Component; }
};

/// Node of the class tree. Each node corresponds to a
/// class definition, a predefined class, or the unnamed root.
class ClassNode {
public:
  ClassNode(std::string name, ClassNode *parent,
            const ast::modelica::ClassDefinition *definition,
            PredefinedClass predefined = PredefinedClass::None);

  llvm::StringRef getName() const { return name; }
  ClassNode *getParent() const { return parent; }
  bool isRoot() const { return parent == nullptr; }

  const ast::modelica::ClassDefinition *getDefinition() const {
    return definition;
  }

  PredefinedClass getPredefined() const { return predefined; }
  bool isPredefined() const { return predefined != PredefinedClass::None; }

  /// Get the names from the root to this class.
  llvm::SmallVector<llvm::StringRef> getPath() const;

  /// Get the path as a dotted string, for diagnostics.
  std::string getPathString() const;

  bool isEnumeration() const;
  bool isEncapsulated() const;

  /// Get the literals of enumerations.
  llvm::ArrayRef<std::string> getEnumerationLiterals() const {
    return literals;
  }

  llvm::ArrayRef<std::unique_ptr<ClassNode>> getChildren() const {
    return children;
  }

  ClassNode *getChild(llvm::StringRef childName) const;

  ClassNode *addChild(std::unique_ptr<ClassNode> child);

  /// Local components, extends-clauses and import-clauses.
  const llvm::StringMap<Element> &getComponents() const { return components; }

  llvm::ArrayRef<const ast::modelica::ExtendsClause *>
  getExtendsClauses() const {
    return extendsClauses;
  }

  llvm::ArrayRef<const ast::modelica::ImportClause *> getImports() const {
    return imports;
  }

  /// Populate the node from its definition, creating the nested classes.
  void populate();

private:
  std::string name;
  ClassNode *parent;
  const ast::modelica::ClassDefinition *definition;
  PredefinedClass predefined;

  std::vector<std::unique_ptr<ClassNode>> children;
  llvm::StringMap<ClassNode *> childrenByName;
  llvm::StringMap<Element> components;
  std::vector<const ast::modelica::ExtendsClause *> extendsClauses;
  std::vector<const ast::modelica::ImportClause *> imports;
  std::vector<std::string> literals;

  friend class ClassTree;
};

/// Class tree built from the parsed files, implementing the lookup of names.
/// Replaceable classes are resolved to their constraining
/// classes, falling back to the declared ones for the
/// elements outside the constraining interface. Replaceable components are
/// resolved to their declared classes.
class ClassTree {
public:
  ClassTree();

  ClassNode *getRoot() { return root.get(); }

  /// Add the classes of a file. Classes of files with a within-clause are
  /// placed within the given package, which must have already been added.
  /// Returns the package that could not be found, if any.
  std::optional<std::string> add(const ast::modelica::StoredDefinition &file);

  /// Look up a simple name starting from the given scope.
  std::optional<Element> lookup(ClassNode *scope, llvm::StringRef name);

  /// Look up a name, possibly composite or global, starting from the given
  /// scope.
  std::optional<Element> lookup(ClassNode *scope,
                                const ast::modelica::Name &name);

  /// Look up an element of a class, including the inherited ones.
  std::optional<Element> lookupMember(ClassNode *classNode,
                                      llvm::StringRef name);

  /// Get the declared class of a component.
  ClassNode *getComponentClass(const Element &component);

  /// Get the base class of a short class definition, that is its
  /// constraining class if the class is replaceable.
  ClassNode *getShortClassBase(ClassNode *classNode);

  /// Get the base classes of a class.
  llvm::SmallVector<ClassNode *> getBaseClasses(ClassNode *classNode);

  /// Resolve a class by its absolute path.
  ClassNode *getClassByPath(llvm::ArrayRef<llvm::StringRef> path);

  /// Get the class an element refers to, following short class definitions
  /// of replaceable classes to their constraining classes.
  ClassNode *getElementClass(const Element &element);

  /// Get the class inherited by a class-extends redeclaration.
  ClassNode *getInheritedClass(ClassNode *classNode);

  /// Resolve the base class of an extends-clause of a class.
  ClassNode *resolveExtends(ClassNode *classNode,
                            const ast::modelica::ExtendsClause &clause);

private:
  std::optional<Element> lookupLocal(ClassNode *classNode,
                                     llvm::StringRef name,
                                     bool includeInherited);

  std::optional<Element> lookupInImports(ClassNode *classNode,
                                         llvm::StringRef name);

  ClassNode *resolveClass(ClassNode *scope, const ast::modelica::Name &name);

private:
  /// Get the node of a class redeclared within the modification of an
  /// extends-clause of the given class.
  ClassNode *
  getRedeclaredClass(const ast::modelica::ClassDefinition &definition,
                     ClassNode *classNode);

private:
  std::unique_ptr<ClassNode> root;

  /// Nodes of the classes redeclared within modifications.
  llvm::DenseMap<const ast::modelica::ClassDefinition *,
                 std::unique_ptr<ClassNode>>
      redeclaredClasses;

  /// Lookups in progress, used to detect cyclic definitions.
  llvm::DenseSet<std::pair<ClassNode *, const void *>> activeLookups;
};
} // namespace marco::codegen::lowering::modelica

#endif // MARCO_CODEGEN_LOWERING_MODELICA_CLASSTREE_H
