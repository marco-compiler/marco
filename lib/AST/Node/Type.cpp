#include "marco/AST/Node/Type.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Record.h"
#include <iostream>
#include <numeric>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const BuiltInType& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(BuiltInType type)
	{
		switch (type) {
			case BuiltInType::None:
				return "None";

			case BuiltInType::Integer:
				return "Integer";

			case BuiltInType::Real:
				return "Real";

			case BuiltInType::String:
				return "String";

			case BuiltInType::Boolean:
				return "Boolean";

			case BuiltInType::Unknown:
				return "Unknown";
		}

    llvm_unreachable("Unexpected type");
    return "";
	}

  BuiltInType getMostGenericNumericBuiltInType(BuiltInType x, BuiltInType y)
  {
    assert(x == BuiltInType::Boolean || x == BuiltInType::Integer || x == BuiltInType::Real);
    assert(y == BuiltInType::Boolean || y == BuiltInType::Integer || y == BuiltInType::Real);

    if (x == y) {
      return x;
    }

    if (x == BuiltInType::Unknown) {
      return y;
    }

    if (y == BuiltInType::Unknown) {
      return x;
    }

    if (x == BuiltInType::Boolean) {
      return y;
    }

    if (y == BuiltInType::Boolean) {
      return x;
    }

    if (x == BuiltInType::Integer) {
      return y;
    }

    return x;
  }

  llvm::Optional<BuiltInType> getMostGenericBuiltInType(BuiltInType x, BuiltInType y)
  {
    if(x == BuiltInType::String && y == BuiltInType::String)
      return x;

    if(x == BuiltInType::String || y == BuiltInType::String)
      return {};
  
    if( x == BuiltInType::Unknown || y == BuiltInType::Unknown)
      return {};

    return getMostGenericNumericBuiltInType(x,y);
  }

  PackedType::PackedType(llvm::ArrayRef<Type> types)
  {
    for (const auto& type : types) {
      this->types.emplace_back(std::make_shared<Type>(type));
    }
  }

  bool PackedType::operator==(const PackedType& other) const
  {
    if (types.size() != other.types.size()) {
      return false;
    }

    auto pairs = llvm::zip(types, other.types);
    return std::all_of(pairs.begin(), pairs.end(),
                       [](const auto& pair)
                       {
                         const auto& [x, y] = pair;
                         return *x == *y;
                       });
  }

  bool PackedType::operator!=(const PackedType& other) const
  {
    return !(*this == other);
  }

  Type& PackedType::operator[](size_t index)
  {
    assert(index < types.size());
    return *types[index];
  }

  Type PackedType::operator[](size_t index) const
  {
    assert(index < types.size());
    return *types[index];
  }

  void PackedType::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << toString(*this);
  }

  bool PackedType::hasConstantShape() const
  {
    return std::all_of(
        types.begin(), types.end(),
        [](const auto& type) {
          return type->hasConstantShape();
        });
  }

  size_t PackedType::size() const { return types.size(); }

  PackedType::iterator PackedType::begin()
  {
    return types.begin();
  }

  PackedType::const_iterator PackedType::begin() const
  {
    return types.begin();
  }

  PackedType::iterator PackedType::end()
  {
    return types.end();
  }

  PackedType::const_iterator PackedType::end() const
  {
    return types.end();
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const PackedType& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(PackedType obj)
  {
    return "{" +
        accumulate(obj.begin(), obj.end(), std::string(),
                   [](const std::string& a, const auto& b) -> std::string {
                     return a + (a.length() > 0 ? ", " : "") + toString(b);
                   }) +
        "}";
  }

  UserDefinedType::UserDefinedType(std::string name, llvm::ArrayRef<Type> types)
      : name(std::move(name))
  {
    for (const auto& type : types) {
      this->types.emplace_back(std::make_shared<Type>(type));
    }
  }

  bool UserDefinedType::operator==(const UserDefinedType& other) const
  {
    if (types.size() != other.types.size()) {
      return false;
    }

    auto pairs = llvm::zip(types, other.types);

    return std::all_of(pairs.begin(), pairs.end(), [](const auto& pair) {
      const auto& [x, y] = pair;
      return *x == *y;
    });
  }

  bool UserDefinedType::operator!=(const UserDefinedType& other) const
  {
    return !(*this == other);
  }

  Type& UserDefinedType::operator[](size_t index)
  {
    assert(index < types.size());
    return *types[index];
  }

  Type UserDefinedType::operator[](size_t index) const
  {
    assert(index < types.size());
    return *types[index];
  }

  void UserDefinedType::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << toString(*this);
  }

  llvm::StringRef UserDefinedType::getName() const
  {
    return name;
  }

  bool UserDefinedType::hasConstantShape() const
  {
    return std::all_of(types.begin(), types.end(), [](const auto& type) {
      return type->hasConstantShape();
    });
  }

  size_t UserDefinedType::size() const
  {
    return types.size();
  }

  UserDefinedType::iterator UserDefinedType::begin()
  {
    return types.begin();
  }

  UserDefinedType::const_iterator UserDefinedType::begin() const
  {
    return types.begin();
  }

  UserDefinedType::iterator UserDefinedType::end()
  {
    return types.end();
  }

  UserDefinedType::const_iterator UserDefinedType::end() const
  {
    return types.end();
  }


  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const UserDefinedType& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(UserDefinedType obj)
  {
    return obj.getName().str() + "{" +
        accumulate(obj.begin(), obj.end(), std::string(),
                   [](const std::string& a, const auto& b) -> std::string {
                     return a + (a.length() > 0 ? ", " : "") + toString(b);
                   }) +
        "}";
  }

  ArrayDimension::ArrayDimension(long size) : size(size)
  {
  }

  ArrayDimension::ArrayDimension(std::unique_ptr<Expression> size)
      : size(std::move(size))
  {
  }

  ArrayDimension::ArrayDimension(const ArrayDimension& other)
  {
    if (other.hasExpression()) {
      size = other.getExpression()->clone();
    } else {
      size = other.getNumericSize();
    }
  }

  ArrayDimension::ArrayDimension(ArrayDimension&& other) = default;

  ArrayDimension::~ArrayDimension() = default;

  ArrayDimension& ArrayDimension::operator=(const ArrayDimension& other)
  {
    ArrayDimension result(other);
    swap(*this, result);
    return *this;
  }

  ArrayDimension& ArrayDimension::operator=(ArrayDimension&& other) = default;

  void swap(ArrayDimension& first, ArrayDimension& second)
  {
    using std::swap;
    swap(first.size, second.size);
  }

  bool ArrayDimension::operator==(const ArrayDimension& other) const
  {
    return size == other.size;
  }

  bool ArrayDimension::operator!=(const ArrayDimension& other) const
  {
    return !(*this == other);
  }

  bool ArrayDimension::hasExpression() const
  {
    return std::holds_alternative<std::unique_ptr<Expression>>(size);
  }

  bool ArrayDimension::isDynamic() const
  {
    return hasExpression() || getNumericSize() == kDynamicSize;
  }

  long ArrayDimension::getNumericSize() const
  {
    assert(std::holds_alternative<long>(size));
    return std::get<long>(size);
  }

  Expression* ArrayDimension::getExpression()
  {
    assert(hasExpression());
    return std::get<std::unique_ptr<Expression>>(size).get();
  }

  const Expression* ArrayDimension::getExpression() const
  {
    assert(hasExpression());
    return std::get<std::unique_ptr<Expression>>(size).get();
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ArrayDimension& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(const ArrayDimension& obj)
  {
    if (obj.hasExpression())
      return toString(*obj.getExpression());

    return std::to_string(obj.getNumericSize());
  }

  Type::Type(BuiltInType type, llvm::ArrayRef<ArrayDimension> dim)
      : content(std::move(type)),
        dimensions(dim.begin(), dim.end())
  {
    assert(std::holds_alternative<BuiltInType>(content));
  }

  Type::Type(PackedType type, llvm::ArrayRef<ArrayDimension> dim)
      : content(std::move(type)),
        dimensions(dim.begin(), dim.end())
  {
    assert(std::holds_alternative<PackedType>(content));
  }

  Type::Type(UserDefinedType type, llvm::ArrayRef<ArrayDimension> dim)
      : content(std::move(type)),
        dimensions(dim.begin(), dim.end())
  {
    assert(std::holds_alternative<UserDefinedType>(content));
  }

  Type::Type(Record *type, llvm::ArrayRef<ArrayDimension> dim)
      : content(std::move(type)),
        dimensions(dim.begin(), dim.end())
  {
    assert(type);
    assert(std::holds_alternative<Record*>(content));
  }

  Type::Type(const Type& other)
      : content(other.content),
        dimensions(other.dimensions.begin(), other.dimensions.end())
  {
  }

  Type::Type(Type&& other) = default;

  Type::~Type() = default;

  Type& Type::operator=(const Type& other)
  {
    Type result(other);
    swap(*this, result);
    return *this;
  }

  Type& Type::operator=(Type&& other) = default;

  void swap(Type& first, Type& second)
  {
    std::swap(first.content, second.content);
    std::swap(first.dimensions, second.dimensions);
  }

  bool Type::operator==(const Type& other) const
  {
    return dimensions == other.dimensions && content == other.content;
  }

  bool Type::operator!=(const Type& other) const
  {
    return !(*this == other);
  }

  ArrayDimension& Type::operator[](int index)
  {
    return dimensions[index];
  }

  const ArrayDimension& Type::operator[](int index) const
  {
    return dimensions[index];
  }

  void Type::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << toString(*this);
  }

  size_t Type::getRank() const
  {
    return dimensions.size();
  }

  llvm::MutableArrayRef<ArrayDimension> Type::getDimensions()
  {
    return dimensions;
  }

  llvm::ArrayRef<ArrayDimension> Type::getDimensions() const
  {
    return dimensions;
  }

  void Type::setDimensions(llvm::ArrayRef<ArrayDimension> dims)
  {
    dimensions.clear();
    dimensions.insert(dimensions.begin(), dims.begin(), dims.end());
  }

  size_t Type::dimensionsCount() const
  {
    return dimensions.size();
  }

  long Type::size() const
  {
    long result = 1;

    for (const auto& dimension : dimensions) {
      if (dimension.hasExpression()) {
        return -1;
      }

      auto numericSize = dimension.getNumericSize();

      if (numericSize == -1) {
        return -1;
      }

      result *= numericSize;
    }

    return result;
  }

  bool Type::hasConstantShape() const
  {
    for (const auto& dimension : dimensions) {
      if (dimension.isDynamic()) {
        return false;
      }
    }

    if (isa<PackedType>()) {
      return get<PackedType>().hasConstantShape();
    }

    return true;
  }

  bool Type::isScalar() const
  {
    return dimensions.empty();
  }

  Type::dimensions_iterator Type::begin()
  {
    return dimensions.begin();
  }

  Type::dimensions_const_iterator Type::begin() const
  {
    return dimensions.begin();
  }

  Type::dimensions_iterator Type::end()
  {
    return dimensions.end();
  }

  Type::dimensions_const_iterator Type::end() const
  {
    return dimensions.end();
  }

  Type Type::subscript(size_t times) const
  {
    assert(!isScalar());

    if (dimensions.size() == times) {
      return visit([&](const auto& t) {
        return Type(t);
      });
    }

    assert(times < dimensions.size());

    return visit([&](const auto& type) {
      return Type(
          type,
          llvm::SmallVector<ArrayDimension, 3>(dimensions.begin() + times, dimensions.end()));
    });
  }

  Type Type::to(BuiltInType type) const
  {
    return Type(type, dimensions);
  }

  Type Type::to(llvm::ArrayRef<ArrayDimension> dims) const
  {
    Type copy = *this;
    copy.setDimensions(dims);
    return copy;
  }

  // helper type for the visitor #4
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  // explicit deduction guide (not needed as of C++20)
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj)
  {
    return stream << toString(obj);
  }

  std::string toString(Type obj)
  {
    auto visitor = overloaded{
        [](const auto& t) { return toString(t); },
        [](Record *&r){ return r->getName().str();}
    };

    auto dimensionsToStringLambda = [](const std::string& a, ArrayDimension& b) -> std::string {
      return a + (a.length() > 0 ? "," : "") +
          (b.hasExpression() ? toString(*b.getExpression()) :
            (b.isDynamic() ? ":" :
              std::to_string(b.getNumericSize())
            )
          );
    };

    auto dimensions = obj.getDimensions();

    std::string size = obj.isScalar() ? ""
                                      : "[" +
            accumulate(
                                            dimensions.begin(),
                                            dimensions.end(),
                                            std::string(),
                                            dimensionsToStringLambda) +
            "]";

    return obj.visit(visitor) + size;
  }

  /**
 *  used when flattening nested arrays of records
 *	e.g. given baseType=Record{NestedRecord[5] a)[10]
 * 			   memberType=NestedRecord[5]
 * 	->	 result = NestedRecord[10,5]
 */
  Type getFlattenedMemberType(Type baseType, Type memberType)
  {
    // note: (todo) concatenating the dimensions should be enough, but it doesn't work since scalars are assumed to be arrays of dimension 1
    if(!baseType.isScalar() )
    {
      if(memberType.isScalar()){
        memberType.setDimensions(baseType.getDimensions());
      }else{
        auto d = memberType.getDimensions();
        llvm::SmallVector<ArrayDimension,3> dimensions(d.begin(),d.end());
        auto new_dimensions = baseType.getDimensions();
        dimensions.insert(dimensions.begin(),new_dimensions.begin(),new_dimensions.end());

        memberType.setDimensions(dimensions);
      }
    }
    return memberType;
  }

  Type Type::unknown()
  {
    return Type(BuiltInType::Unknown);
  }

  FunctionType::FunctionType(llvm::ArrayRef<Type> args, llvm::ArrayRef<Type> results)
  {
    for (const auto& type : args) {
      this->args.push_back(type);
    }

    for (const auto& type : results) {
      this->results.push_back(type);
    }
  }

  FunctionType::FunctionType(const FunctionType& other)
      : args(other.args),
        results(other.results)
  {
  }

  FunctionType::FunctionType(FunctionType&& other) = default;

  FunctionType::~FunctionType() = default;

  FunctionType& FunctionType::operator=(const FunctionType& other)
  {
    FunctionType result(other);
    swap(*this, result);
    return *this;
  }

  FunctionType& FunctionType::operator=(FunctionType&& other) = default;

  void swap(FunctionType& first, FunctionType& second)
  {
    std::swap(first.args, second.args);
    std::swap(first.results, second.results);
  }

  void FunctionType::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents) << "function type";
    os.indent(indents + 1) << "args:";

    for (const auto& type : args) {
      type.dump(os, indents + 2);
    }

    os << "\n";
    os.indent(indents + 1) << "results:";

    for (const auto& type : results) {
      type.dump(os, indents + 2);
    }
  }

  llvm::ArrayRef<Type> FunctionType::getArgs() const
  {
    return args;
  }

  llvm::ArrayRef<Type> FunctionType::getResults() const
  {
    return results;
  }

  Type FunctionType::packResults() const
  {
    if (results.size() == 1)
      return results[0];

    return Type(PackedType(results));
  }
}
