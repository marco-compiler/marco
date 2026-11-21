#include "marco/Codegen/Lowering/Modelica/ClassTree.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::codegen::lowering::modelica;

//===---------------------------------------------------------------------===//
// ClassNode
//===---------------------------------------------------------------------===//

namespace marco::codegen::lowering::modelica {
ClassNode::ClassNode(std::string name, ClassNode *parent,
                     const ClassDefinition *definition,
                     PredefinedClass predefined)
    : name(std::move(name)), parent(parent), definition(definition),
      predefined(predefined) {}

llvm::SmallVector<llvm::StringRef> ClassNode::getPath() const {
  llvm::SmallVector<llvm::StringRef> path;

  for (const ClassNode *node = this; node && !node->isRoot();
       node = node->parent) {
    path.push_back(node->name);
  }

  std::reverse(path.begin(), path.end());
  return path;
}

std::string ClassNode::getPathString() const {
  return llvm::join(getPath(), ".");
}

bool ClassNode::isEnumeration() const {
  if (predefined == PredefinedClass::StateSelect ||
      predefined == PredefinedClass::AssertionLevel) {
    return true;
  }

  return definition &&
         definition->getForm() == ClassDefinition::Form::Enumeration;
}

bool ClassNode::isEncapsulated() const {
  return definition && definition->isEncapsulated();
}

ClassNode *ClassNode::getChild(llvm::StringRef childName) const {
  auto it = childrenByName.find(childName);
  return it == childrenByName.end() ? nullptr : it->second;
}

ClassNode *ClassNode::addChild(std::unique_ptr<ClassNode> child) {
  if (childrenByName.contains(child->getName())) {
    return nullptr;
  }

  ClassNode *result = child.get();
  childrenByName[child->getName()] = result;
  children.push_back(std::move(child));
  return result;
}

void ClassNode::populate() {
  if (!definition) {
    return;
  }

  for (const Child<EnumerationLiteral> &literal :
       definition->getEnumerationLiterals()) {
    literals.push_back(literal->getName().str());
  }

  for (const Child<Node> &element : definition->getElements()) {
    if (auto classDefinition = element->dyn_cast<ClassDefinition>()) {
      auto child = std::make_unique<ClassNode>(
          classDefinition->getName().str(), this, classDefinition);

      if (ClassNode *added = addChild(std::move(child))) {
        added->populate();
      }

      continue;
    }

    if (auto clause = element->dyn_cast<ComponentClause>()) {
      for (const Child<ComponentDeclaration> &declaration :
           clause->getDeclarations()) {
        Element component;
        component.kind = Element::Kind::Component;
        component.clause = clause;
        component.declaration = declaration.get();
        component.owner = this;
        components.try_emplace(declaration->getName(), component);
      }

      continue;
    }

    if (auto extendsClause = element->dyn_cast<ExtendsClause>()) {
      extendsClauses.push_back(extendsClause);
      continue;
    }

    if (auto importClause = element->dyn_cast<ImportClause>()) {
      imports.push_back(importClause);
    }
  }
}
} // namespace marco::codegen::lowering::modelica

//===---------------------------------------------------------------------===//
// ClassTree
//===---------------------------------------------------------------------===//

namespace {
/// Tags identifying the kinds of lookup in progress.
const char inheritedLookupTag = 0;
const char shortClassTag = 0;
const char inheritedClassTag = 0;
} // namespace

namespace marco::codegen::lowering::modelica {
ClassTree::ClassTree()
    : root(std::make_unique<ClassNode>("", nullptr, nullptr)) {
  auto addPredefined = [&](llvm::StringRef name, PredefinedClass kind,
                           std::vector<std::string> literals = {}) {
    auto node =
        std::make_unique<ClassNode>(name.str(), root.get(), nullptr, kind);

    node->literals = std::move(literals);
    root->addChild(std::move(node));
  };

  // Predefined types.
  addPredefined("Real", PredefinedClass::Real);
  addPredefined("Integer", PredefinedClass::Integer);
  addPredefined("Boolean", PredefinedClass::Boolean);
  addPredefined("String", PredefinedClass::String);

  addPredefined("StateSelect", PredefinedClass::StateSelect,
                {"never", "avoid", "default", "prefer", "always"});

  addPredefined("AssertionLevel", PredefinedClass::AssertionLevel,
                {"error", "warning"});

  addPredefined("ExternalObject", PredefinedClass::ExternalObject);
  addPredefined("Connections", PredefinedClass::Connections);
  addPredefined("Clock", PredefinedClass::Clock);
}

std::optional<std::string> ClassTree::add(const StoredDefinition &file) {
  ClassNode *target = root.get();

  if (const auto &within = file.getWithin(); within && !within->parts.empty()) {
    llvm::SmallVector<llvm::StringRef> path;

    for (const std::string &part : within->parts) {
      path.push_back(part);
    }

    target = getClassByPath(path);

    if (!target) {
      return within->toString();
    }
  }

  for (const Child<ClassDefinition> &classDefinition : file.getClasses()) {
    auto child = std::make_unique<ClassNode>(
        classDefinition->getName().str(), target, classDefinition.get());

    if (ClassNode *added = target->addChild(std::move(child))) {
      added->populate();
    }
  }

  return std::nullopt;
}

ClassNode *ClassTree::getClassByPath(llvm::ArrayRef<llvm::StringRef> path) {
  ClassNode *node = root.get();

  for (llvm::StringRef part : path) {
    node = node->getChild(part);

    if (!node) {
      return nullptr;
    }
  }

  return node;
}

std::optional<Element> ClassTree::lookupLocal(ClassNode *classNode,
                                              llvm::StringRef name,
                                              bool includeInherited) {
  if (!classNode) {
    return std::nullopt;
  }

  const ClassDefinition *definition = classNode->getDefinition();

  // The elements of a short class definition are the ones of its base class.
  if (definition && definition->getForm() == ClassDefinition::Form::Short) {
    if (!includeInherited) {
      return std::nullopt;
    }

    if (auto element = lookupLocal(getShortClassBase(classNode), name, true)) {
      return element;
    }

    // Libraries commonly refer to elements of replaceable classes that are
    // not part of the constraining interface, in which case the declared
    // class is used.
    if (definition->getConstrainingClause()) {
      return lookupLocal(resolveClass(classNode->getParent(),
                                      definition->getBaseClass()),
                         name, true);
    }

    return std::nullopt;
  }

  if (ClassNode *child = classNode->getChild(name)) {
    Element element;
    element.kind = Element::Kind::Class;
    element.classNode = child;
    element.owner = classNode;
    return element;
  }

  if (auto it = classNode->getComponents().find(name);
      it != classNode->getComponents().end()) {
    return it->second;
  }

  if (!includeInherited) {
    return std::nullopt;
  }

  std::pair<ClassNode *, const void *> key(classNode, &inheritedLookupTag);

  if (!activeLookups.insert(key).second) {
    return std::nullopt;
  }

  std::optional<Element> result;

  for (const ExtendsClause *clause : classNode->getExtendsClauses()) {
    ClassNode *baseClass = resolveExtends(classNode, *clause);

    if (!baseClass || !(result = lookupLocal(baseClass, name, true))) {
      continue;
    }

    // Inherited elements are accessed through the most derived class.
    result->accessScope = classNode;

    // Elements redeclared by the modification of the extends-clause
    // are replaced by the new declaration, whose names are
    // resolved in the scope of the modification.
    if (clause->getClassModification()) {
      for (const Child<Argument> &argument :
           clause->getClassModification()->getArguments()) {
        auto redeclaration = argument->dyn_cast<ElementRedeclaration>();

        if (!redeclaration || !redeclaration->isRedeclare()) {
          continue;
        }

        if (auto componentClause =
                redeclaration->getElement()->dyn_cast<ComponentClause>();
            componentClause && result->isComponent() &&
            componentClause->getDeclarations()[0]->getName() == name) {
          result->clause = componentClause;
          result->declaration = componentClause->getDeclarations()[0].get();
          result->owner = classNode;
          result->accessScope = nullptr;
        }

        if (auto classDefinition =
                redeclaration->getElement()->dyn_cast<ClassDefinition>();
            classDefinition && result->isClass() &&
            classDefinition->getName() == name) {
          result->classNode = getRedeclaredClass(*classDefinition, classNode);
          result->owner = classNode;
          result->accessScope = nullptr;
        }
      }
    }

    break;
  }

  // A class-extends redeclaration inherits from the class it redeclares.
  if (!result && classNode->getDefinition() &&
      classNode->getDefinition()->getForm() ==
          ClassDefinition::Form::ClassExtends) {
    if ((result = lookupLocal(getInheritedClass(classNode), name, true))) {
      result->accessScope = classNode;
    }
  }

  activeLookups.erase(key);
  return result;
}

std::optional<Element> ClassTree::lookupInImports(ClassNode *classNode,
                                                  llvm::StringRef name) {
  for (const ImportClause *importClause : classNode->getImports()) {
    const Name &importedName = importClause->getName();

    switch (importClause->getForm()) {
    case ImportClause::Form::Qualified:
      if (importedName.parts.back() == name) {
        return lookup(root.get(), importedName);
      }

      break;

    case ImportClause::Form::Renaming:
      if (importClause->getAlias() == name) {
        return lookup(root.get(), importedName);
      }

      break;

    case ImportClause::Form::Unqualified:
      if (ClassNode *package = resolveClass(root.get(), importedName)) {
        if (auto element = lookupLocal(package, name, true)) {
          return element;
        }
      }

      break;

    case ImportClause::Form::Multiple:
      if (llvm::is_contained(importClause->getImportedNames(), name)) {
        if (ClassNode *package = resolveClass(root.get(), importedName)) {
          return lookupLocal(package, name, true);
        }
      }

      break;
    }
  }

  return std::nullopt;
}

std::optional<Element> ClassTree::lookup(ClassNode *scope,
                                         llvm::StringRef name) {
  for (ClassNode *classNode = scope; classNode;
       classNode = classNode->getParent()) {
    if (auto element = lookupLocal(classNode, name, true)) {
      return element;
    }

    if (auto element = lookupInImports(classNode, name)) {
      return element;
    }

    // Lookup within encapsulated classes stops at their boundary, except for
    // the predefined classes.
    if (classNode->isEncapsulated()) {
      if (ClassNode *predefined = root->getChild(name);
          predefined && predefined->isPredefined()) {
        Element element;
        element.classNode = predefined;
        element.owner = root.get();
        return element;
      }

      return std::nullopt;
    }
  }

  return std::nullopt;
}

std::optional<Element> ClassTree::lookup(ClassNode *scope, const Name &name) {
  if (name.parts.empty()) {
    return std::nullopt;
  }

  std::optional<Element> element =
      name.global ? lookupLocal(root.get(), name.parts[0], false)
                  : lookup(scope, name.parts[0]);

  for (size_t i = 1; element && i < name.parts.size(); ++i) {
    element = lookupMember(getElementClass(*element), name.parts[i]);
  }

  return element;
}

std::optional<Element> ClassTree::lookupMember(ClassNode *classNode,
                                               llvm::StringRef name) {
  return lookupLocal(classNode, name, true);
}

ClassNode *ClassTree::resolveClass(ClassNode *scope, const Name &name) {
  auto element = lookup(scope, name);

  if (!element || !element->isClass()) {
    return nullptr;
  }

  return element->classNode;
}

ClassNode *ClassTree::getElementClass(const Element &element) {
  if (element.isComponent()) {
    return getComponentClass(element);
  }

  return element.classNode;
}

ClassNode *ClassTree::getComponentClass(const Element &component) {
  // The declared class of replaceable components is used, rather than the
  // constraining one: being a subtype of the constraining class, it leads to
  // the same types of values, while allowing the access to the elements not
  // belonging to the constraining interface, which libraries commonly rely
  // upon.
  const Name &typeName = component.clause->getTypeSpecifier();

  // The types of inherited components are resolved in the derived class
  // first, so that its redeclarations of replaceable classes are used. This
  // approximates the lookup in the instance tree.
  if (component.accessScope && component.accessScope != component.owner &&
      !typeName.global) {
    if (ClassNode *result = resolveClass(component.accessScope, typeName)) {
      return result;
    }
  }

  return resolveClass(component.owner, typeName);
}

ClassNode *ClassTree::getShortClassBase(ClassNode *classNode) {
  const ClassDefinition *definition = classNode->getDefinition();

  if (!definition || definition->getForm() != ClassDefinition::Form::Short) {
    return nullptr;
  }

  std::pair<ClassNode *, const void *> key(classNode, &shortClassTag);

  if (!activeLookups.insert(key).second) {
    return nullptr;
  }

  // The base class is looked up in the scope enclosing the definition.
  const Name &baseName = definition->getConstrainingClause()
                             ? definition->getConstrainingClause()
                                   ->getTypeSpecifier()
                             : definition->getBaseClass();

  ClassNode *result = resolveClass(classNode->getParent(), baseName);
  activeLookups.erase(key);
  return result;
}

ClassNode *ClassTree::resolveExtends(ClassNode *classNode,
                                     const ExtendsClause &clause) {
  std::pair<ClassNode *, const void *> key(classNode, &clause);

  if (!activeLookups.insert(key).second) {
    return nullptr;
  }

  // The base class is looked up without considering the inherited elements
  // of the class, as the result must not depend on them.
  const Name &name = clause.getBaseClass();
  std::optional<Element> element;

  if (name.global) {
    element = lookupLocal(root.get(), name.parts[0], false);
  } else {
    element = lookupLocal(classNode, name.parts[0], false);

    if (!element) {
      element = lookupInImports(classNode, name.parts[0]);
    }

    if (!element && !classNode->isEncapsulated()) {
      element = lookup(classNode->getParent(), name.parts[0]);
    }
  }

  for (size_t i = 1; element && i < name.parts.size(); ++i) {
    element = lookupMember(getElementClass(*element), name.parts[i]);
  }

  activeLookups.erase(key);

  if (!element || !element->isClass()) {
    return nullptr;
  }

  return element->classNode;
}

ClassNode *ClassTree::getInheritedClass(ClassNode *classNode) {
  ClassNode *parent = classNode->getParent();

  if (!parent) {
    return nullptr;
  }

  std::pair<ClassNode *, const void *> key(classNode, &inheritedClassTag);

  if (!activeLookups.insert(key).second) {
    return nullptr;
  }

  ClassNode *result = nullptr;

  for (const ExtendsClause *clause : parent->getExtendsClauses()) {
    if (ClassNode *baseClass = resolveExtends(parent, *clause)) {
      if (auto element = lookupLocal(baseClass, classNode->getName(), true);
          element && element->isClass()) {
        result = element->classNode;
        break;
      }
    }
  }

  activeLookups.erase(key);
  return result;
}

ClassNode *ClassTree::getRedeclaredClass(const ClassDefinition &definition,
                                         ClassNode *classNode) {
  std::unique_ptr<ClassNode> &node = redeclaredClasses[&definition];

  if (!node) {
    // The redeclared class is resolved in the scope of the class whose
    // extends-clause contains the modification.
    node = std::make_unique<ClassNode>(definition.getName().str(), classNode,
                                       &definition);

    node->populate();
  }

  return node.get();
}

llvm::SmallVector<ClassNode *> ClassTree::getBaseClasses(ClassNode *classNode) {
  llvm::SmallVector<ClassNode *> result;

  for (const ExtendsClause *clause : classNode->getExtendsClauses()) {
    if (ClassNode *baseClass = resolveExtends(classNode, *clause)) {
      result.push_back(baseClass);
    }
  }

  // A class-extends redeclaration inherits from the class it redeclares.
  if (const ClassDefinition *definition = classNode->getDefinition();
      definition &&
      definition->getForm() == ClassDefinition::Form::ClassExtends) {
    if (ClassNode *inherited = getInheritedClass(classNode)) {
      result.push_back(inherited);
    }
  }

  return result;
}
} // namespace marco::codegen::lowering::modelica
