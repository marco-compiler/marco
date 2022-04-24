#include "marco/AST/Node/RecordInstance.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Record.h"
#include "marco/AST/Node/Member.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  RecordInstance::RecordInstance(SourceRange location, Type type, llvm::ArrayRef<std::unique_ptr<Expression>> values)
      : ASTNode(std::move(location)),type(type)
  {
    setType(std::move(type));

    for (const auto& v : values) {
      this->values.push_back(v->clone());
    }
  }

  RecordInstance::RecordInstance(const RecordInstance& other)
      : ASTNode(other),
        recordType(other.recordType),
        type(other.type)
  {
    for (const auto& v : other.values) {
      values.push_back(v->clone());
    }
  }

  RecordInstance::RecordInstance(RecordInstance&& other) = default;

  RecordInstance::~RecordInstance()  = default;

  RecordInstance& RecordInstance::operator=(const RecordInstance& other)
  {
    RecordInstance result(other);
    swap(*this, result);
    return *this;
  }

  RecordInstance& RecordInstance::operator=(RecordInstance&& other) = default;

  bool RecordInstance::operator==(const RecordInstance& other) const
  {
    if (recordType != other.recordType) {
      return false;
    }

    if (values.size() != other.values.size()) {
      return false;
    }

    auto pairs = llvm::zip(values, other.values);

    return std::all_of(pairs.begin(), pairs.end(), [](const auto& pair) {
      const auto& [x, y] = pair;
      return *x == *y;
    });
  }

  bool RecordInstance::operator!=(const RecordInstance& other) const
  {
    return !(*this == other);
  }

  bool RecordInstance::isLValue() const
  {
    return false;
  }

  void RecordInstance::setType(Type tp)
  {
    assert(type.isa<Record*>());
    assert(type.get<Record*>()!=nullptr);

    type = std::move(tp);
    recordType = type.get<Record*>();
  }

  Type& RecordInstance::getType()
  {
    return type;
  }

  const Type& RecordInstance::getType() const
  {
    return type;
  }

  const Record* RecordInstance::getRecordType() const
  {
    return recordType;
  }

  RecordInstance::iterator RecordInstance::begin()
  {
    return values.begin();
  }

  RecordInstance::const_iterator RecordInstance::begin() const
  {
    return values.begin();
  }

  RecordInstance::iterator RecordInstance::end()
  {
    return values.end();
  }

  RecordInstance::const_iterator RecordInstance::end() const
  {
    return values.end();
  }

  void swap(RecordInstance& first, RecordInstance& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    impl::swap(first.values, second.values);
    swap(first.type, second.type);
    swap(first.recordType, second.recordType);
  }

  void RecordInstance::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "record instance (" << recordType->getName() << ") : \n";

    for (const auto& v : values) {
      v->print(os, indents + 1);
    }
  }

  Expression& RecordInstance::getMemberValue(llvm::StringRef name)
  {
    auto it = values.begin();

    for (const auto& m : *recordType) {
      if (m->getName() == name){
        return **it;
      }

      ++it;
    }

    assert(false && "not enough values in record instance");
    return **it;
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(const Record& obj)
  {
    return obj.getName().str() + "(" +
        std::accumulate(obj.begin(), obj.end(), std::string(),
                   [](const std::string& result, const std::unique_ptr<Member>& member)
                   {
                     std::string str = toString(member->getType()) + " " + member->getName().str();
                     return result.empty() ? str : result + "," + str;
                   }) +
        ")";
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const RecordInstance& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(const RecordInstance& obj)
  {
    return "RecordInstance<" +
        obj.getRecordType()->getName().str() + ">(" +
        accumulate(obj.begin(), obj.end(), std::string(),
                   [](const std::string& result, const std::unique_ptr<Expression>& e)
                   {
                     std::string str = toString(*e);
                     return result.empty() ? str : result + "," + str;
                   }) +
        ")";
  }
}
