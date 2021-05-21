#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace modelica::frontend
{
	class Type;
	class Expression;

	enum class BuiltInType
	{
		None,
		Integer,
		Float,
		String,
		Boolean,
		Unknown
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const BuiltInType& obj);

	std::string toString(BuiltInType type);

	template<typename T>
	constexpr BuiltInType typeToFrontendType()
	{
		if constexpr (std::is_same<T, float>() || std::is_same<T, double>())
			return BuiltInType::Float;
		if constexpr (std::is_same<T, int>() || std::is_same<T, long>())
			return BuiltInType::Integer;
		if constexpr (std::is_same<T, bool>())
			return BuiltInType::Boolean;
		if constexpr (std::is_same<std::string, T>())
			return BuiltInType::String;

		assert(false && "Unknown type");
		return BuiltInType::Unknown;
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
		using value = long;
	};

	template<>
	class frontendTypeToType<BuiltInType::Float>
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

	/**
	 * Represent the size of an array dimension.
	 * Can be either static or determined by an expression. Note that
	 * a dynamic size (":", in Modelica) is considered static and is
	 * represented by value "-1".
	 */
	class ArrayDimension
	{
		public:
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

		Type(BuiltInType type, llvm::ArrayRef<ArrayDimension> dim = { 1 });
		Type(PackedType type, llvm::ArrayRef<ArrayDimension> dim = { 1 });
		Type(UserDefinedType type, llvm::ArrayRef<ArrayDimension> dim = { 1 });

		Type(const Type& other);
		Type(Type&& other);

		~Type();

		Type& operator=(const Type& other);
		Type& operator=(Type&& other);

		friend void swap(Type& first, Type& second);

		[[nodiscard]] bool operator==(const Type& other) const;
		[[nodiscard]] bool operator!=(const Type& other) const;

		[[nodiscard]] ArrayDimension& operator[](int index);
		[[nodiscard]] const ArrayDimension& operator[](int index) const;

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
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

		[[nodiscard]] llvm::MutableArrayRef<ArrayDimension> getDimensions();
		[[nodiscard]] llvm::ArrayRef<ArrayDimension> getDimensions() const;
		void setDimensions(llvm::ArrayRef<ArrayDimension> dimensions);

		[[nodiscard]] size_t dimensionsCount() const;
		[[nodiscard]] size_t size() const;

		[[nodiscard]] bool hasConstantShape() const;

		[[nodiscard]] bool isScalar() const;

		[[nodiscard]] dimensions_iterator begin();
		[[nodiscard]] dimensions_const_iterator begin() const;

		[[nodiscard]] dimensions_iterator end();
		[[nodiscard]] dimensions_const_iterator end() const;

		[[nodiscard]] Type subscript(size_t times) const;

		[[nodiscard]] Type to(BuiltInType type);

		[[nodiscard]] static Type unknown();

		private:
		std::variant<BuiltInType, PackedType, UserDefinedType> content;
		llvm::SmallVector<ArrayDimension, 3> dimensions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj);

	std::string toString(Type obj);

	template<typename T, typename... Args>
	[[nodiscard]] Type makeType(Args&&... args)
	{
		static_assert(typeToFrontendType<T>() != BuiltInType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(typeToFrontendType<T>());

		return Type(typeToFrontendType<T>(), { static_cast<ArrayDimension>(std::forward<Args>(args))... });
	}

	template<BuiltInType T, typename... Args>
	[[nodiscard]] Type makeType(Args&&... args)
	{
		static_assert(T != BuiltInType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(T);

		return Type(T, { static_cast<ArrayDimension>(std::forward<Args>(args))... });
	}
}
