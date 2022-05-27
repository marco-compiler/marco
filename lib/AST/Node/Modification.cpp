#include "marco/AST/Node/Modification.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Modification::Modification(
      SourceRange location,
      std::unique_ptr<ClassModification> classModification)
    : ASTNode(std::move(location)),
      classModification(std::move(classModification)),
      expression(llvm::None)
  {
  }

  Modification::Modification(
      SourceRange location,
      std::unique_ptr<ClassModification> classModification,
      std::unique_ptr<Expression> expression)
    : ASTNode(std::move(location)),
      classModification(std::move(classModification)),
      expression(std::move(expression))
  {
  }

  Modification::Modification(
      SourceRange location,
      std::unique_ptr<Expression> expression)
    : ASTNode(std::move(location)),
      classModification(llvm::None),
      expression(std::move(expression))
  {
  }

  Modification::Modification(const Modification& other)
    : ASTNode(other)
  {
    if (other.classModification.hasValue()) {
      classModification = other.classModification.getValue()->clone();
    } else {
      classModification = llvm::None;
    }

    if (other.expression.hasValue()) {
      expression = other.expression.getValue()->clone();
    } else {
      expression = llvm::None;
    }
  }

  Modification::Modification(Modification&& other) = default;

  Modification::~Modification() = default;

  Modification& Modification::operator=(const Modification& other)
  {
    Modification result(other);
    swap(*this, result);
    return *this;
  }

  Modification& Modification::operator=(Modification&& other) = default;

  void swap(Modification& first, Modification& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.classModification, second.classModification);
    swap(first.expression, second.expression);
  }

  void Modification::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents) << "modification:\n";

    if (classModification.hasValue())
      (*classModification)->dump(os, indents + 1);

    if (expression.hasValue())
      (*expression)->dump(os, indents + 1);
  }

  bool Modification::hasClassModification() const
  {
    return classModification.hasValue();
  }

  ClassModification* Modification::getClassModification()
  {
    assert(hasClassModification());
    return classModification->get();
  }

  const ClassModification* Modification::getClassModification() const
  {
    assert(hasClassModification());
    return classModification->get();
  }

  bool Modification::hasExpression() const
  {
    return expression.hasValue();
  }

  Expression* Modification::getExpression()
  {
    assert(hasExpression());
    return expression->get();
  }

  const Expression* Modification::getExpression() const
  {
    assert(hasExpression());
    return expression->get();
  }

  ClassModification::ClassModification(
      SourceRange location,
      llvm::ArrayRef<std::unique_ptr<Argument>> arguments)
    : ASTNode(std::move(location))
  {
    for (const auto& arg : arguments) {
      this->arguments.push_back(arg->clone());
    }
  }

  ClassModification::ClassModification(const ClassModification& other)
    : ASTNode(other)
  {
    for (const auto& arg : other.arguments) {
      this->arguments.push_back(arg->clone());
    }
  }

  ClassModification::ClassModification(ClassModification&& other) = default;

  ClassModification::~ClassModification() = default;

  ClassModification& ClassModification::operator=(const ClassModification& other)
  {
    ClassModification result(other);
    swap(*this, result);
    return *this;
  }

  ClassModification& ClassModification::operator=(ClassModification&& other) = default;

  void swap(ClassModification& first, ClassModification& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    impl::swap(first.arguments, second.arguments);
  }

  void ClassModification::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents) << "class-modification:\n";

    for (const auto& argument : arguments) {
      argument->dump(os, indents + 1);
    }
  }

  ClassModification::iterator ClassModification::begin()
  {
    return arguments.begin();
  }

  ClassModification::const_iterator ClassModification::begin() const
  {
    return arguments.begin();
  }

  ClassModification::iterator ClassModification::end()
  {
    return arguments.end();
  }

  ClassModification::const_iterator ClassModification::end() const
  {
    return arguments.end();
  }

  Argument::Argument(ElementModification content)
      : content(std::move(content))
  {
  }

  Argument::Argument(ElementRedeclaration content)
      : content(std::move(content))
  {
  }

  Argument::Argument(ElementReplaceable content)
      : content(std::move(content))
  {
  }

  Argument::Argument(const Argument& other)
      : content(other.content)
  {
  }

  Argument::Argument(Argument&& other) = default;

  Argument::~Argument() = default;

  Argument& Argument::operator=(const Argument& other)
  {
    Argument result(other);
    swap(*this, result);
    return *this;
  }

  Argument& Argument::operator=(Argument&& other) = default;

  void swap(Argument& first, Argument& second)
  {
    using std::swap;
    swap(first.content, second.content);
  }

  void Argument::print(llvm::raw_ostream& os, size_t indents) const
  {
    visit([&os, indents](const auto& obj) {
      obj.print(os, indents);
    });
  }

  ElementModification::ElementModification(
      SourceRange location,
      bool each,
      bool final,
      llvm::StringRef name,
    std::unique_ptr<Modification> modification)
    : ASTNode(std::move(location)),
      each(each),
      final(final),
      name(name.str()),
      modification(std::move(modification))
  {
  }

  ElementModification::ElementModification(
      SourceRange location,
      bool each,
      bool final,
      llvm::StringRef name)
    : ASTNode(std::move(location)),
      each(each),
      final(final),
      name(name.str())
  {
  }

  ElementModification::ElementModification(const ElementModification& other)
      : ASTNode(other),
        each(other.each),
        final(other.final),
        name(other.name)
  {
    if (other.modification.hasValue()) {
      modification = other.modification.getValue()->clone();
    } else {
      modification = llvm::None;
    }
  }

  ElementModification::ElementModification(ElementModification&& other) = default;

  ElementModification::~ElementModification() = default;

  ElementModification& ElementModification::operator=(const ElementModification& other)
  {
    ElementModification result(other);
    swap(*this, result);
    return *this;
  }

  ElementModification& ElementModification::operator=(ElementModification&& other) = default;

  void swap(ElementModification& first, ElementModification& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.each, second.each);
    swap(first.final, second.final);
    swap(first.name, second.name);
    swap(first.modification, second.modification);
  }

  void ElementModification::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents) << "element-modification:\n";

    if (each) {
      os.indent(indents + 1) << "each: " << each << "\n";
    }

    if (final) {
      os.indent(indents + 1) << "final: " << final << "\n";
    }

    os.indent(indents + 1) << "name: " << name << "\n";

    if (modification.hasValue()) {
      (*modification)->dump(os, indents + 1);
    }
  }

  bool ElementModification::hasEachProperty() const
  {
    return each;
  }

  bool ElementModification::hasFinalProperty() const
  {
    return final;
  }

  llvm::StringRef ElementModification::getName() const
  {
    return name;
  }

  bool ElementModification::hasModification() const
  {
    return modification.hasValue();
  }

  Modification* ElementModification::getModification()
  {
    assert(hasModification());
    return modification->get();
  }

  const Modification* ElementModification::getModification() const
  {
    assert(hasModification());
    return modification->get();
  }

  ElementReplaceable::ElementReplaceable(SourceRange location)
      : ASTNode(std::move(location))
  {
  }

  ElementReplaceable::ElementReplaceable(const ElementReplaceable& other)
      : ASTNode(other)
  {
  }

  ElementReplaceable::ElementReplaceable(ElementReplaceable&& other) = default;

  ElementReplaceable::~ElementReplaceable() = default;

  ElementReplaceable& ElementReplaceable::operator=(const ElementReplaceable& other)
  {
    ElementReplaceable result(other);
    swap(*this, result);
    return *this;
  }

  ElementReplaceable& ElementReplaceable::operator=(ElementReplaceable&& other) = default;

  void swap(ElementReplaceable& first, ElementReplaceable& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
  }

  void ElementReplaceable::print(llvm::raw_ostream& os, size_t indents) const
  {
    // TODO
  }

  ElementRedeclaration::ElementRedeclaration(SourceRange location)
      : ASTNode(std::move(location))
  {
  }

  ElementRedeclaration::ElementRedeclaration(const ElementRedeclaration& other)
      : ASTNode(other)
  {
  }

  ElementRedeclaration::ElementRedeclaration(ElementRedeclaration&& other) = default;

  ElementRedeclaration::~ElementRedeclaration() = default;

  ElementRedeclaration& ElementRedeclaration::operator=(const ElementRedeclaration& other)
  {
    ElementRedeclaration result(other);
    swap(*this, result);
    return *this;
  }

  ElementRedeclaration& ElementRedeclaration::operator=(ElementRedeclaration&& other) = default;

  void swap(ElementRedeclaration& first, ElementRedeclaration& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
  }

  void ElementRedeclaration::print(llvm::raw_ostream& os, size_t indents) const
  {
    // TODO
  }
}
