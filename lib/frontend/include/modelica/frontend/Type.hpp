#pragma once

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

	enum class BuiltinType
	{
		None,
		Integer,
		Float,
		String,
		Boolean,
		Unknown
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const BuiltinType& obj);

	std::string toString(BuiltinType type);

	template<typename T>
	constexpr BuiltinType typeToFrontendType()
	{
		if constexpr (std::is_same<T, double>::value)
			return BuiltinType::Float;
		if constexpr (std::is_same<T, int>::value)
			return BuiltinType::Integer;
		if constexpr (std::is_same<T, bool>::value)
			return BuiltinType::Boolean;
		if constexpr (std::is_same<std::string, T>::value)
			return BuiltinType::String;

		assert(false && "Unknown type");
		return BuiltinType::Unknown;
	}

	template<BuiltinType T>
	class frontendTypeToType;

	template<>
	class frontendTypeToType<BuiltinType::Boolean>
	{
		public:
		using value = bool;
	};

	template<>
	class frontendTypeToType<BuiltinType::Float>
	{
		public:
		using value = double;
	};

	template<>
	class frontendTypeToType<BuiltinType::Integer>
	{
		public:
		using value = int;
	};

	template<>
	class frontendTypeToType<BuiltinType::String>
	{
		public:
		using value = std::string;
	};

	template<BuiltinType T>
	using frontendTypeToType_v = typename frontendTypeToType<T>::value;

	class UserDefinedType
	{
		public:
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

		[[nodiscard]] llvm::SmallVectorImpl<UniqueType>::const_iterator begin()
				const;
		[[nodiscard]] llvm::SmallVectorImpl<UniqueType>::const_iterator end() const;

		private:
		llvm::SmallVector<UniqueType, 3> types;
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const UserDefinedType& obj);

	std::string toString(UserDefinedType obj);

	class Type
	{
		public:
		Type(BuiltinType type, llvm::ArrayRef<size_t> dim = { 1 });
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
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		[[nodiscard]] llvm::SmallVectorImpl<size_t>& getDimensions();
		[[nodiscard]] const llvm::SmallVectorImpl<size_t>& getDimensions() const;

		[[nodiscard]] size_t dimensionsCount() const;
		[[nodiscard]] size_t size() const;

		[[nodiscard]] bool isScalar() const;

		[[nodiscard]] llvm::SmallVectorImpl<size_t>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<size_t>::const_iterator begin() const;

		[[nodiscard]] llvm::SmallVectorImpl<size_t>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<size_t>::const_iterator end() const;

		[[nodiscard]] Type subscript(size_t times) const;

		[[nodiscard]] static Type Int();
		[[nodiscard]] static Type Float();
		[[nodiscard]] static Type unknown();

		private:
		std::variant<BuiltinType, UserDefinedType> content;
		llvm::SmallVector<size_t, 3> dimensions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj);

	std::string toString(Type obj);

	template<typename T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		static_assert(typeToFrontendType<T>() != BuiltinType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(typeToFrontendType<T>());

		return Type(typeToFrontendType<T>(), { static_cast<size_t>(args)... });
	}

	template<BuiltinType T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		static_assert(T != BuiltinType::Unknown);

		if constexpr (sizeof...(Args) == 0)
			return Type(T);

		return Type(T, { static_cast<size_t>(args)... });
	}

}	 // namespace modelica
