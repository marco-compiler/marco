#ifndef MARCO_AST_NODE_TYPE_H
#define MARCO_AST_NODE_TYPE_H

#include "marco/AST/Node/ASTNode.h"
#include "boost/iterator/indirect_iterator.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace marco::ast
{
  class Type;
  class Expression;
  class Record;

  enum class BuiltInType
  {
    None,
    Integer,
    Real,
    String,
    Boolean,
    Unknown
  };

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const BuiltInType& obj);

  std::string toString(BuiltInType type);

  /// Get the most generic built-in numeric type.
  ///     x/y     Unknown       Boolean       Integer       Real
  ///   Unknown     x/y           y              y           y
  ///   Boolean      x           x/y             y           y
  ///   Integer      x            x             x/y          y
  ///   Real         x            x              x          x/y
  BuiltInType getMostGenericBuiltInType(BuiltInType x, BuiltInType y);

  namespace detail
  {
    template<typename T>
    struct [[maybe_unused]] IsBuiltInTypeCompatible
    {
      static const bool value = false;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<bool>
    {
      static const bool value = true;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<int>
    {
      static const bool value = true;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<long>
    {
      static const bool value = true;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<float>
    {
      static const bool value = true;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<double>
    {
      static const bool value = true;
    };

    template<>
    struct [[maybe_unused]] IsBuiltInTypeCompatible<std::string>
    {
      static const bool value = true;
    };

    template<typename T>
    struct [[maybe_unused]] FrontendType;

    template<>
    struct [[maybe_unused]] FrontendType<bool>
    {
      static const BuiltInType value = BuiltInType::Boolean;
    };

    template<>
    struct [[maybe_unused]] FrontendType<int>
    {
      static const BuiltInType value = BuiltInType::Integer;
    };

    template<>
    struct [[maybe_unused]] FrontendType<long>
    {
      static const BuiltInType value = BuiltInType::Integer;
    };

    template<>
    struct [[maybe_unused]] FrontendType<float>
    {
      static const BuiltInType value = BuiltInType::Real;
    };

    template<>
    struct [[maybe_unused]] FrontendType<double>
    {
      static const BuiltInType value = BuiltInType::Real;
    };

    template<>
    struct [[maybe_unused]] FrontendType<std::string>
    {
      static const BuiltInType value = BuiltInType::String;
    };
  }

  template<BuiltInType T>
  class frontendTypeToType;

  template<>
  class frontendTypeToType<BuiltInType::Boolean>
  {
    public:
      using value = bool;
  };

  template<>
  class frontendTypeToType<BuiltInType::Integer>
  {
    public:
      using value = int64_t;
  };

  template<>
  class frontendTypeToType<BuiltInType::Real>
  {
    public:
      using value = double;
  };

  template<>
  class frontendTypeToType<BuiltInType::String>
  {
    public:
      using value = std::string;
  };

  template<BuiltInType T>
  using frontendTypeToType_v = typename frontendTypeToType<T>::value;

  class PackedType
  {
    private:
      using TypePtr = std::shared_ptr<Type>;
      using Container = llvm::SmallVector<TypePtr, 3>;

    public:
      using iterator = boost::indirect_iterator<Container::iterator>;
      using const_iterator = boost::indirect_iterator<Container::const_iterator>;

      explicit PackedType(llvm::ArrayRef<Type> types);

      [[nodiscard]] bool operator==(const PackedType& other) const;
      [[nodiscard]] bool operator!=(const PackedType& other) const;

      [[nodiscard]] Type& operator[](size_t index);
      [[nodiscard]] Type operator[](size_t index) const;

      void print(llvm::raw_ostream& os, size_t indents = 0) const;

      [[nodiscard]] bool hasConstantShape() const;

      [[nodiscard]] size_t size() const;

      [[nodiscard]] iterator begin();
      [[nodiscard]] const_iterator begin() const;

      [[nodiscard]] iterator end();
      [[nodiscard]] const_iterator end() const;

    private:
      Container types;
  };

  llvm::raw_ostream& operator<<(
          llvm::raw_ostream& stream, const PackedType& obj);

  std::string toString(PackedType obj);

  class UserDefinedType : public impl::Dumpable<UserDefinedType>
  {
    private:
      using TypePtr = std::shared_ptr<Type>;
      using Container = llvm::SmallVector<TypePtr, 3>;

    public:
      using iterator = boost::indirect_iterator<Container::iterator>;
      using const_iterator = boost::indirect_iterator<Container::const_iterator>;

      UserDefinedType(std::string name, llvm::ArrayRef<Type> types);

      [[nodiscard]] bool operator==(const UserDefinedType& other) const;
      [[nodiscard]] bool operator!=(const UserDefinedType& other) const;

      [[nodiscard]] Type& operator[](size_t index);
      [[nodiscard]] Type operator[](size_t index) const;

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] llvm::StringRef getName() const;

      [[nodiscard]] bool hasConstantShape() const;

      [[nodiscard]] size_t size() const;

      [[nodiscard]] iterator begin();
      [[nodiscard]] const_iterator begin() const;

      [[nodiscard]] iterator end();
      [[nodiscard]] const_iterator end() const;

    private:
      std::string name;
      Container types;
  };

  llvm::raw_ostream& operator<<(
          llvm::raw_ostream& stream, const UserDefinedType& obj);

  std::string toString(UserDefinedType obj);

  /// Represent the size of an array dimension.
  /// Can be either static or determined by an expression. Note that
  /// a dynamic size (":", in Modelica) is considered static and is
  /// represented by value "-1".
  class ArrayDimension
  {
    public:
      static constexpr long kDynamicSize = -1;

      ArrayDimension(long size);
      ArrayDimension(std::unique_ptr<Expression> size);

      ArrayDimension(const ArrayDimension& other);
      ArrayDimension(ArrayDimension&& other);

      ~ArrayDimension();

      ArrayDimension& operator=(const ArrayDimension& other);
      ArrayDimension& operator=(ArrayDimension&& other);

      friend void swap(ArrayDimension& first, ArrayDimension& second);

      [[nodiscard]] bool operator==(const ArrayDimension& other) const;
      [[nodiscard]] bool operator!=(const ArrayDimension& other) const;

      template<class Visitor>
      auto visit(Visitor&& visitor)
      {
        return std::visit(std::forward<Visitor>(visitor), size);
      }

      template<class Visitor>
      auto visit(Visitor&& visitor) const
      {
        return std::visit(std::forward<Visitor>(visitor), size);
      }

      [[nodiscard]] bool hasExpression() const;

      [[nodiscard]] bool isDynamic() const;

      [[nodiscard]] long getNumericSize() const;

      [[nodiscard]] Expression* getExpression();
      [[nodiscard]] const Expression* getExpression() const;

    private:
      std::variant<long, std::unique_ptr<Expression>> size;
  };

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ArrayDimension& obj);

  std::string toString(const ArrayDimension& obj);

  class Type : public impl::Dumpable<Type>
  {
    private:
      template <typename T> using Container = llvm::SmallVector<T, 3>;

    public:
      using dimensions_iterator = Container<ArrayDimension>::iterator;
      using dimensions_const_iterator = Container<ArrayDimension>::const_iterator;

      Type(BuiltInType type, llvm::ArrayRef<ArrayDimension> dim = llvm::None);
      Type(PackedType type, llvm::ArrayRef<ArrayDimension> dim = llvm::None);
      Type(UserDefinedType type, llvm::ArrayRef<ArrayDimension> dim = llvm::None);
      Type(Record *type, llvm::ArrayRef<ArrayDimension> dim = llvm::None);

      Type(const Type& other);
      Type(Type&& other);

      ~Type() override;

      Type& operator=(const Type& other);
      Type& operator=(Type&& other);

      friend void swap(Type& first, Type& second);

      [[nodiscard]] bool operator==(const Type& other) const;
      [[nodiscard]] bool operator!=(const Type& other) const;

      [[nodiscard]] ArrayDimension& operator[](int index);
      [[nodiscard]] const ArrayDimension& operator[](int index) const;

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      template<BuiltInType T>
      [[nodiscard]] bool isa() const
      {
        if (!std::holds_alternative<BuiltInType>(content))
          return false;

        return std::get<BuiltInType>(content) == T;
      }

      template<typename T,  typename std::enable_if<
          !detail::IsBuiltInTypeCompatible<T>::value, bool>::type = true>
      [[nodiscard]] bool isa() const
      {
        return std::holds_alternative<T>(content);
      }

      template<typename T, typename std::enable_if<
          detail::IsBuiltInTypeCompatible<T>::value, bool>::type = true>
      [[nodiscard]] bool isa() const
      {
        if (!std::holds_alternative<BuiltInType>(content))
            return false;

        return detail::FrontendType<T>::value == std::get<BuiltInType>(content);
      }

      [[nodiscard]] bool isNumeric() const
      {
        return isa<bool>() || isa<int>() || isa<float>();
      }

      template<typename T>
      [[nodiscard]] T& get()
      {
        assert(isa<T>());
        return std::get<T>(content);
      }

      template<typename T>
      [[nodiscard]] const T& get() const
      {
        assert(isa<T>());
        return std::get<T>(content);
      }

      template<class Visitor>
      auto visit(Visitor&& visitor)
      {
        return std::visit(std::forward<Visitor>(visitor), content);
      }

      template<class Visitor>
      auto visit(Visitor&& visitor) const
      {
        return std::visit(std::forward<Visitor>(visitor), content);
      }

      [[nodiscard]] size_t getRank() const;

      [[nodiscard]] llvm::MutableArrayRef<ArrayDimension> getDimensions();
      [[nodiscard]] llvm::ArrayRef<ArrayDimension> getDimensions() const;
      void setDimensions(llvm::ArrayRef<ArrayDimension> dimensions);

      [[nodiscard]] size_t dimensionsCount() const;
      [[nodiscard]] long size() const;

      [[nodiscard]] bool hasConstantShape() const;

      [[nodiscard]] bool isScalar() const;

      [[nodiscard]] dimensions_iterator begin();
      [[nodiscard]] dimensions_const_iterator begin() const;

      [[nodiscard]] dimensions_iterator end();
      [[nodiscard]] dimensions_const_iterator end() const;

      [[nodiscard]] Type subscript(size_t times) const;

      [[nodiscard]] Type to(BuiltInType type) const;
      [[nodiscard]] Type to(llvm::ArrayRef<ArrayDimension> dimensions) const;

      [[nodiscard]] static Type unknown();

    private:
      std::variant<BuiltInType, PackedType, UserDefinedType, Record*> content;
      llvm::SmallVector<ArrayDimension, 3> dimensions;
  };

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj);

  std::string toString(Type obj);

  Type getFlattenedMemberType(Type baseType, Type memberType);

  template<typename T, typename... Args>
  [[nodiscard]] Type makeType(Args&&... args)
  {
    static_assert(detail::IsBuiltInTypeCompatible<T>::value);

    if constexpr (sizeof...(Args) == 0)
        return Type(detail::FrontendType<T>::value);

    return Type(detail::FrontendType<T>::value, { static_cast<ArrayDimension>(std::forward<Args>(args))... });
  }

  template<BuiltInType T, typename... Args>
  [[nodiscard]] Type makeType(Args&&... args)
  {
    static_assert(T != BuiltInType::Unknown);

    if constexpr (sizeof...(Args) == 0)
        return Type(T);

    return Type(T, { static_cast<ArrayDimension>(std::forward<Args>(args))... });
  }

  class FunctionType : public impl::Dumpable<FunctionType>
  {
    public:
      FunctionType(llvm::ArrayRef<Type> args, llvm::ArrayRef<Type> results);

      FunctionType(const FunctionType& other);
      FunctionType(FunctionType&& other);

      ~FunctionType() override;

      FunctionType& operator=(const FunctionType& other);
      FunctionType& operator=(FunctionType&& other);

      friend void swap(FunctionType& first, FunctionType& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] llvm::ArrayRef<Type> getArgs() const;
      [[nodiscard]] llvm::ArrayRef<Type> getResults() const;

      [[nodiscard]] Type packResults() const;

    private:
      llvm::SmallVector<Type, 3> args;
      llvm::SmallVector<Type, 1> results;
  };
}

#endif // MARCO_AST_NODE_TYPE_H
