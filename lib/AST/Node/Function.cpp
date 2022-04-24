#include "marco/AST/AST.h"
#include <algorithm>
#include <variant>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Function::Function(SourceRange location, llvm::StringRef name)
      : ASTNode(std::move(location)),
        name(name.str())
  {
  }

  Function::Function(const Function& other)
      : ASTNode(other),
        name(other.name)
  {
  }

  Function::Function(Function&& other) = default;

  Function::~Function() = default;

  Function& Function::operator=(const Function& other)
  {
    if (this != &other) {
      static_cast<ASTNode&>(*this) = static_cast<const ASTNode&>(other);

      this->name = other.name;
    }

    return *this;
  }

  Function& Function::operator=(Function&& other) = default;

  void swap(Function& first, Function& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.name, second.name);
  }

  llvm::StringRef Function::getName() const
  {
    return name;
  }

  PartialDerFunction::PartialDerFunction(
      SourceRange location,
      llvm::StringRef name,
      std::unique_ptr<Expression> derivedFunction,
      llvm::ArrayRef<std::unique_ptr<Expression>> independentVariables)
    : Function(std::move(location), name),
      derivedFunction(std::move(derivedFunction))
  {
    for (const auto& var : independentVariables) {
      this->independentVariables.push_back(var->clone());
    }
  }

  PartialDerFunction::PartialDerFunction(const PartialDerFunction& other)
    : Function(other),
      derivedFunction(other.derivedFunction->clone())
  {
    independentVariables.clear();

    for (const auto& arg : other.independentVariables) {
      independentVariables.push_back(arg->clone());
    }

    args.clear();

    for (const auto& type : other.args) {
      args.push_back(type);
    }

    results.clear();

    for (const auto& type : other.results) {
      results.push_back(type);
    }
  }

  PartialDerFunction::PartialDerFunction(PartialDerFunction&& other) = default;

  PartialDerFunction::~PartialDerFunction() = default;

  PartialDerFunction& PartialDerFunction::operator=(const PartialDerFunction& other)
  {
    PartialDerFunction result(other);
    swap(*this, result);
    return *this;
  }

  PartialDerFunction& PartialDerFunction::operator=(PartialDerFunction&& other) = default;

  void swap(PartialDerFunction& first, PartialDerFunction& second)
  {
    swap(static_cast<Function&>(first), static_cast<Function&>(second));

    using std::swap;
    swap(first.derivedFunction, second.derivedFunction);
    impl::swap(first.independentVariables, second.independentVariables);
    swap(first.args, second.args);
    swap(first.results, second.results);
  }

  void PartialDerFunction::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "function " << getName() << "\n";

    os.indent(indents + 1);
    os << "derived function\n";
    derivedFunction->dump(os, indents + 2);

    os.indent(indents + 1);
    os << "args\n";

    for (const auto& var : independentVariables) {
      var->dump(os, indents + 2);
    }
  }

  Expression* PartialDerFunction::getDerivedFunction() const
  {
    return derivedFunction.get();
  }

  llvm::MutableArrayRef<std::unique_ptr<Expression>> PartialDerFunction::getIndependentVariables()
  {
    return independentVariables;
  }

  llvm::ArrayRef<std::unique_ptr<Expression>> PartialDerFunction::getIndependentVariables() const
  {
    return independentVariables;
  }

  llvm::MutableArrayRef<Type> PartialDerFunction::getArgsTypes()
  {
    return args;
  }

  llvm::ArrayRef<Type> PartialDerFunction::getArgsTypes() const
  {
    return args;
  }

  void PartialDerFunction::setArgsTypes(llvm::ArrayRef<Type> types)
  {
    this->args.clear();

    for (const auto& type : types) {
      this->args.push_back(type);
    }
  }

  llvm::MutableArrayRef<Type> PartialDerFunction::getResultsTypes()
  {
    return results;
  }

  llvm::ArrayRef<Type> PartialDerFunction::getResultsTypes() const
  {
    return results;
  }

  void PartialDerFunction::setResultsTypes(llvm::ArrayRef<Type> types)
  {
    this->results.clear();

    for (const auto& type : types) {
      this->results.push_back(type);
    }
  }

  FunctionType PartialDerFunction::getType() const
  {
    return FunctionType(args, results);
  }

  StandardFunction::StandardFunction(
      SourceRange location,
      bool pure,
      llvm::StringRef name,
      llvm::ArrayRef<std::unique_ptr<Member>> members,
      llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
      llvm::Optional<std::unique_ptr<Annotation>> annotation)
    : Function(std::move(location), name),
      pure(pure)
  {
    for (const auto& member : members) {
      this->members.push_back(member->clone());
    }

    for (const auto& algorithm : algorithms) {
      this->algorithms.push_back(algorithm->clone());
    }

    if (annotation.hasValue()) {
      this->annotation = annotation.getValue()->clone();
    } else {
      this->annotation = llvm::None;
    }
  }

  StandardFunction::StandardFunction(const StandardFunction& other)
    : Function(other),
      pure(other.pure)
  {
    for (const auto& member : other.members) {
      this->members.push_back(member->clone());
    }

    for (const auto& algorithm : other.algorithms) {
      this->algorithms.push_back(algorithm->clone());
    }

    if (other.annotation.hasValue()) {
      annotation = other.annotation.getValue()->clone();
    } else {
      annotation = llvm::None;
    }
  }

  StandardFunction::StandardFunction(StandardFunction&& other) = default;

  StandardFunction::~StandardFunction() = default;

  StandardFunction& StandardFunction::operator=(const StandardFunction& other)
  {
    StandardFunction result(other);
    swap(*this, result);
    return *this;
  }

  StandardFunction& StandardFunction::operator=(StandardFunction&& other) = default;

  void swap(StandardFunction& first, StandardFunction& second)
  {
    swap(static_cast<Function&>(first), static_cast<Function&>(second));

    using std::swap;
    swap(first.pure, second.pure);
    impl::swap(first.members, second.members);
    impl::swap(first.algorithms, second.algorithms);
    swap(first.annotation, second.annotation);
  }

  void StandardFunction::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "function " << getName() << "\n";

    for (const auto& member : members) {
      member->print(os, indents + 1);
    }

    for (const auto& algorithm : getAlgorithms()) {
      algorithm->print(os, indents + 1);
    }
  }

  Member* StandardFunction::operator[](llvm::StringRef name)
  {
    for (auto& member : members) {
      if (member->getName() == name) {
        return member.get();
      }
    }

    assert(false && "Not found");
    return nullptr;
  }

  const Member* StandardFunction::operator[](llvm::StringRef name) const
  {
    for (const auto& member : members) {
      if (member->getName() == name) {
        return member.get();
      }
    }

    assert(false && "Not found");
    return nullptr;
  }

  bool StandardFunction::isPure() const
  {
    return pure;
  }

  llvm::MutableArrayRef<std::unique_ptr<Member>> StandardFunction::getMembers()
  {
    return members;
  }

  llvm::ArrayRef<std::unique_ptr<Member>> StandardFunction::getMembers() const
  {
    return members;
  }

  StandardFunction::Container<Member*> StandardFunction::getArgs() const
  {
    Container<Member*> result;

    for (const auto& member : members) {
      if (member->isInput()) {
        result.push_back(member.get());
      }
    }

    return result;
  }

  StandardFunction::Container<Member*> StandardFunction::getResults() const
  {
    Container<Member*> result;

    for (const auto& member : members) {
      if (member->isOutput()) {
        result.push_back(member.get());
      }
    }

    return result;
  }

  StandardFunction::Container<Member*> StandardFunction::getProtectedMembers() const
  {
    Container<Member*> result;

    for (const auto& member : members) {
      if (!member->isInput() && !member->isOutput()) {
        result.push_back(member.get());
      }
    }

    return result;
  }

  void StandardFunction::addMember(std::unique_ptr<Member> member)
  {
    this->members.push_back(std::move(member));
  }

  llvm::MutableArrayRef<std::unique_ptr<Algorithm>> StandardFunction::getAlgorithms()
  {
    return algorithms;
  }

  llvm::ArrayRef<std::unique_ptr<Algorithm>> StandardFunction::getAlgorithms() const
  {
    return algorithms;
  }

  bool StandardFunction::hasAnnotation() const
  {
    return annotation.hasValue();
  }

  Annotation* StandardFunction::getAnnotation()
  {
    assert(annotation.hasValue());
    return annotation.getValue().get();
  }

  const Annotation* StandardFunction::getAnnotation() const
  {
    assert(annotation.hasValue());
    return annotation.getValue().get();
  }

  bool StandardFunction::shouldBeInlined() const
  {
    return hasAnnotation() ? getAnnotation()->getInlineProperty() : false;
  }

  bool StandardFunction::isCustomRecordConstructor() const
  {
    auto str = getName().str();

    return str.find(".constructor.") != std::string::npos ||
        str.find(".'constructor'.") != std::string::npos;
  }

  FunctionType StandardFunction::getType() const
  {
    llvm::SmallVector<Type, 3> argsTypes;
    llvm::SmallVector<Type, 1> resultsTypes;

    for (const auto& member : members) {
      if (member->isInput()) {
        argsTypes.push_back(member->getType());
      } else if (member->isOutput()) {
        resultsTypes.push_back(member->getType());
      }
    }

    return FunctionType(argsTypes, resultsTypes);
  }

  DerivativeAnnotation::DerivativeAnnotation(llvm::StringRef name, unsigned int order)
    : name(name.str()), order(order)
  {
    assert(order > 0);
  }

  llvm::StringRef DerivativeAnnotation::getName() const
  {
    return name;
  }

  unsigned int DerivativeAnnotation::getOrder() const
  {
    return order;
  }

  InverseFunctionAnnotation::InverseFunctionAnnotation() = default;

  bool InverseFunctionAnnotation::isInvertible(llvm::StringRef arg) const
  {
    return map.find(arg) != map.end();
  }

  llvm::StringRef InverseFunctionAnnotation::getInverseFunction(llvm::StringRef invertibleArg) const
  {
    assert(isInvertible(invertibleArg));
    return map.find(invertibleArg)->second.first;
  }

  llvm::ArrayRef<std::string> InverseFunctionAnnotation::getInverseArgs(llvm::StringRef invertibleArg) const
  {
    assert(isInvertible(invertibleArg));
    return map.find(invertibleArg)->second.second;
  }

  void InverseFunctionAnnotation::addInverse(llvm::StringRef invertedArg, llvm::StringRef inverseFunctionName, llvm::ArrayRef<std::string> args)
  {
    assert(map.find(invertedArg) == map.end());
    Container<std::string> c(args.begin(), args.end());
    map[invertedArg] = std::make_pair(inverseFunctionName.str(), c);
  }
}
