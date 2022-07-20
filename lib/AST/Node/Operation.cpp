#include "marco/AST/Node/Operation.h"
#include "marco/AST/Node/Expression.h"
#include "llvm/ADT/StringRef.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const OperationKind& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(OperationKind operation)
	{
		switch (operation) {
			case OperationKind::negate:
				return "negate";

			case OperationKind::add:
				return "add";

      case OperationKind::addEW:
        return "addEW";

			case OperationKind::subtract:
				return "subtract";

      case OperationKind::subtractEW:
        return "subtractEW";

			case OperationKind::multiply:
				return "multiply";

      case OperationKind::multiplyEW:
        return "multiplyEW";

			case OperationKind::divide:
				return "divide";

      case OperationKind::divideEW:
        return "divideEW";

			case OperationKind::ifelse:
				return "ifelse";

			case OperationKind::greater:
				return "greater";

			case OperationKind::greaterEqual:
				return "greaterEqual";

			case OperationKind::equal:
				return "equal";

			case OperationKind::different:
				return "different";

			case OperationKind::lessEqual:
				return "lessEqual";

			case OperationKind::less:
				return "less";

			case OperationKind::land:
				return "land";

      case OperationKind::lnot:
        return "lnot";

			case OperationKind::lor:
				return "lor";

			case OperationKind::subscription:
				return "subscription";

			case OperationKind::memberLookup:
				return "memberLookup";

			case OperationKind::powerOf:
				return "powerOf";

      case OperationKind::powerOfEW:
        return "powerOfEW";

			case OperationKind::range:
				return "range";
		}

		return "unexpected";
	}

  Operation::Operation(
      SourceRange location,
      Type type,
      OperationKind kind,
      llvm::ArrayRef<std::unique_ptr<Expression>> args)
    : ASTNode(std::move(location)),
      type(std::move(type)),
      kind(kind)
  {
    for (const auto& arg : args) {
      this->args.push_back(arg->clone());
    }
  }

  Operation::Operation(const Operation& other)
    : ASTNode(other),
      type(other.type),
      kind(other.kind)
  {
    for (const auto& arg : other.args) {
      this->args.push_back(arg->clone());
    }
  }

  Operation::Operation(Operation&& other) = default;

  Operation::~Operation() = default;

  Operation& Operation::operator=(const Operation& other)
  {
    Operation result(other);
    swap(*this, result);
    return *this;
  }

  Operation& Operation::operator=(Operation&& other) = default;

  void swap(Operation& first, Operation& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.type, second.type);
    swap(first.kind, second.kind);
    impl::swap(first.args, second.args);
  }

  void Operation::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "operation kind: " << getOperationKind() << "\n";

    os.indent(indents);
    os << "type: ";
    getType().print(os);
    os << "\n";

    os.indent(indents);
    os << "args:\n";

    for (const auto& arg : getArguments()) {
      arg->print(os, indents + 1);
    }
  }

  bool Operation::isLValue() const
  {
    switch (getOperationKind()) {
      case OperationKind::subscription:
        return args[0]->isLValue();

      case OperationKind::memberLookup:
        return true;

      default:
        return false;
    }
  }

  bool Operation::operator==(const Operation& other) const
  {
    if (kind != other.kind) {
      return false;
    }

    if (args.size() != other.args.size()) {
      return false;
    }

    return args == other.args;
  }

  bool Operation::operator!=(const Operation& other) const
  {
    return !(*this == other);
  }

  Expression* Operation::operator[](size_t index)
  {
    return getArg(index);
  }

  const Expression* Operation::operator[](size_t index) const
  {
    return getArg(index);
  }

  Type& Operation::getType()
  {
    return type;
  }

  const Type& Operation::getType() const
  {
    return type;
  }

  void Operation::setType(Type tp)
  {
    type = std::move(tp);
  }

  OperationKind Operation::getOperationKind() const
  {
    return kind;
  }

  void Operation::setOperationKind(OperationKind newKind)
  {
    this->kind = newKind;
  }

  Expression* Operation::getArg(size_t index)
  {
    assert(index < args.size());
    return args[index].get();
  }

  const Expression* Operation::getArg(size_t index) const
  {
    assert(index < args.size());
    return args[index].get();
  }

  llvm::MutableArrayRef<std::unique_ptr<Expression>> Operation::getArguments()
  {
    return args;
  }

  llvm::ArrayRef<std::unique_ptr<Expression>> Operation::getArguments() const
  {
    return args;
  }

  size_t Operation::argumentsCount() const
  {
    return args.size();
  }

  size_t Operation::size() const
  {
    return args.size();
  }

  Operation::iterator Operation::begin()
  {
    return args.begin();
  }

  Operation::const_iterator Operation::begin() const
  {
    return args.begin();
  }

  Operation::iterator Operation::end()
  {
    return args.end();
  }

  Operation::const_iterator Operation::end() const
  {
    return args.end();
  }

  void Operation::removeArg(size_t index)
  {
    assert(!args.empty() && index<args.size() && "invalid index");
    args.erase(args.begin()+index);
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Operation& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(const Operation& obj)
  {
    switch (obj.getOperationKind())
    {
      case OperationKind::negate:
        return "(not" + toString(*obj[0]) + ")";

      case OperationKind::add:
      case OperationKind::addEW:
        return "(" +
            accumulate(obj.begin(), obj.end(), std::string(),
                       [](const std::string& result, const std::unique_ptr<Expression>& element) {
                         std::string str = toString(*element);
                         return result.empty() ? str : result + " + " + str;
                       })
            + ")";

      case OperationKind::subtract:
      case OperationKind::subtractEW:
        return "(" +
            accumulate(obj.begin(), obj.end(), std::string(),
                       [](const std::string& result, const std::unique_ptr<Expression>& element) {
                         std::string str = toString(*element);
                         return result.empty() ? str : result + " - " + str;
                       })
            + ")";

      case OperationKind::multiply:
      case OperationKind::multiplyEW:
        return "(" +
            accumulate(obj.begin(), obj.end(), std::string(),
                       [](const std::string& result, const std::unique_ptr<Expression>& element) {
                         std::string str = toString(*element);
                         return result.empty() ? str : result + " * " + str;
                       })
            + ")";

      case OperationKind::divide:
      case OperationKind::divideEW:
        return "(" +
            accumulate(obj.begin(), obj.end(), std::string(),
                       [](const std::string& result, const std::unique_ptr<Expression>& element) {
                         std::string str = toString(*element);
                         return result.empty() ? str : result + " / " + str;
                       })
            + ")";

      case OperationKind::ifelse:
        return "(" + toString(*obj[0]) + " ? " + toString(*obj[1]) + " : " + toString(*obj[2]) + ")";

      case OperationKind::greater:
        return "(" + toString(*obj[0]) + " > " + toString(*obj[1]) + ")";

      case OperationKind::greaterEqual:
        return "(" + toString(*obj[0]) + " >= " + toString(*obj[1]) + ")";

      case OperationKind::equal:
        return "(" + toString(*obj[0]) + " == " + toString(*obj[1]) + ")";

      case OperationKind::different:
        return "(" + toString(*obj[0]) + " != " + toString(*obj[1]) + ")";

      case OperationKind::lessEqual:
        return "(" + toString(*obj[0]) + " <= " + toString(*obj[1]) + ")";

      case OperationKind::less:
        return "(" + toString(*obj[0]) + " < " + toString(*obj[1]) + ")";

      case OperationKind::land:
        return "(" + toString(*obj[0]) + " && " + toString(*obj[1]) + ")";

      case OperationKind::lnot:
        return "(not " + toString(*obj[0]) + ")";

      case OperationKind::lor:
        return "(" + toString(*obj[0]) + " || " + toString(*obj[1]) + ")";

      case OperationKind::subscription:
        return "(" + toString(*obj[0]) +
            accumulate(std::next(obj.begin()), obj.end(), std::string(),
                       [](const std::string& result, const std::unique_ptr<Expression>& element) {
                         std::string str = toString(*element);
                         return result + "[" + str + "]";
                       }) +
            ")";

      case OperationKind::memberLookup:
        return "(" + toString(*obj[0]) + " . " + toString(*obj[1]) + ")";

      case OperationKind::powerOf:
      case OperationKind::powerOfEW:
        return "(" + toString(*obj[0]) + " ^ " + toString(*obj[1]) + ")";

      case OperationKind::range:
        if (obj.argumentsCount() == 3) {
          return toString(*obj[0]) + " : " + toString(*obj[1]) + " : " + toString(*obj[2]);
        }

        return toString(*obj[0]) + " : " + toString(*obj[1]);
    }

    return "unknown";
  }
}
