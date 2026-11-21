#ifndef MARCO_AST_MODELICA_CLASSDEFINITION_H
#define MARCO_AST_MODELICA_CLASSDEFINITION_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/ConstrainingClause.h"
#include "marco/AST/Modelica/Description.h"
#include "marco/AST/Modelica/ElementPrefixes.h"
#include "marco/AST/Modelica/EnumerationLiteral.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/ExternalClause.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/TypePrefix.h"

namespace marco::ast::modelica {
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

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ElementPrefixes &getPrefixes();

  const ElementPrefixes &getPrefixes() const;

  bool isEncapsulated() const;

  void setEncapsulated(bool value);

  bool isPartial() const;

  void setPartial(bool value);

  Purity getPurity() const;

  void setPurity(Purity value);

  ClassRestriction getRestriction() const;

  void setRestriction(ClassRestriction value);

  llvm::StringRef getName() const;

  void setName(std::string newName);

  Form getForm() const;

  void setForm(Form value);

  /// Description string and annotation. For long class definitions, the
  /// annotation is the one at the end of the composition.
  Description &getDescription();

  const Description &getDescription() const;

  /// @name Long and class-extends definitions
  /// {

  /// Get the elements and the sections, in declaration order. Elements are
  /// ClassDefinition, ComponentClause, ExtendsClause and ImportClause nodes,
  /// while sections are EquationSection and AlgorithmSection nodes.
  ChildrenRef<Node> getElements() const;

  void addElement(std::unique_ptr<Node> node);

  ExternalClause *getExternalClause() const;

  void setExternalClause(std::unique_ptr<ExternalClause> node);

  /// }
  /// @name Class-extends and short definitions
  /// {

  /// Modification of the inherited class (class-extends) or of the base
  /// class (short definitions).
  ClassModification *getClassModification() const;

  void setClassModification(std::unique_ptr<ClassModification> node);

  /// }
  /// @name Short definitions
  /// {

  Causality getBasePrefix() const;

  void setBasePrefix(Causality value);

  const Name &getBaseClass() const;

  void setBaseClass(Name name);

  ChildrenRef<Expression> getSubscripts() const;

  void addSubscript(std::unique_ptr<Expression> node);

  /// }
  /// @name Enumerations
  /// {

  ChildrenRef<EnumerationLiteral> getEnumerationLiterals() const;

  void addEnumerationLiteral(std::unique_ptr<EnumerationLiteral> node);

  /// Whether the enumeration is `enumeration(:)`.
  bool isUnspecifiedEnumeration() const;

  void setUnspecifiedEnumeration(bool value);

  /// }
  /// @name Derivative definitions
  /// {

  const Name &getDerivedFunction() const;

  void setDerivedFunction(Name name);

  llvm::ArrayRef<std::string> getDerivativeArguments() const;

  void addDerivativeArgument(std::string arg);

  /// }

  ConstrainingClause *getConstrainingClause() const;

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node);

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
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CLASSDEFINITION_H
