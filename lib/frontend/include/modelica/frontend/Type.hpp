#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace modelica
{
	class Type;
	using UniqueType = std::unique_ptr<Type>;

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
		using value = int;
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

	class UserDefinedType
	{
		private:
		using Container = llvm::SmallVector<UniqueType, 3>;

		public:
		using iterator = boost::indirect_iterator<Container::iterator>;
		using const_iterator = boost::indirect_iterator<Container::const_iterator>;

		explicit UserDefinedType(llvm::ArrayRef<Type> types);

		UserDefinedType(const UserDefinedType& other);
		UserDefinedType(UserDefinedType&& other) = default;

		~UserDefinedType() = default;

		UserDefinedType& operator=(const UserDefinedType& other);
		UserDefinedType& operator=(UserDefinedType&& other) = default;

		[[nodiscard]] bool operator==(const UserDefinedType& other) const;
		[[nodiscard]] bool operator!=(const UserDefinedType& other) const;

		[[nodiscard]] Type& operator[](size_t index);
		[[nodiscard]] Type operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container types;
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const UserDefinedType& obj);

	std::string toString(UserDefinedType obj);

	class Type
	{
		public:
		using dimensions_iterator = llvm::SmallVectorImpl<size_t>::iterator;
		using dimensions_const_iterator = llvm::SmallVectorImpl<size_t>::const_iterator;

		Type(BuiltInType type, llvm::ArrayRef<size_t> dim = { 1 });
		Type(UserDefinedType type, llvm::ArrayRef<size_t> dim = { 1 });
		Type(llvm::ArrayRef<Type> members, llvm::ArrayRef<size_t> dim = { 1 });

		[[nodiscard]] bool operator==(const Type& other) const;
		[[nodiscard]] bool operator!=(const Type& other) const;

		[[nodiscard]] size_t& operator[](int index);
		[[nodiscard]] size_t operator[](int index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

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
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		[[nodiscard]] llvm::SmallVectorImpl<size_t>& getDimensions();
		[[nodiscard]] const llvm::SmallVectorImpl<size_t>& getDimensions() const;

		[[nodiscard]] size_t dimensionsCount() const;
		[[nodiscard]] size_t size() const;

		[[nodiscard]] bool isScalar() const;

		[[nodiscard]] dimensions_iterator begin();
		[[nodiscard]] dimensions_const_iterator begin() const;

		[[nodiscard]] dimensions_iterator end();
		[[nodiscard]] dimensions_const_iterator end() const;

		[[nodiscard]] Type subscript(size_t times) const;

		[[nodiscard]] static Type Bool();
		[[nodiscard]] static Type Int();
		[[nodiscard]] static Type Float();
		[[nodiscard]] static Type unknown();

		private:
		std::variant<BuiltInType, UserDefinedType> content;
		llvm::SmallVector<size_t, 3> dimensions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj);

	std::string toString(Type obj);

	template<typename T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		static_assert(typeToFrontendType<T>() != BuiltInType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(typeToFrontendType<T>());

		return Type(typeToFrontendType<T>(), { static_cast<size_t>(args)... });
	}

	template<BuiltInType T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		static_assert(T != BuiltInType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(T);

		return Type(T, { static_cast<size_t>(args)... });
	}

}	 // namespace modelica
