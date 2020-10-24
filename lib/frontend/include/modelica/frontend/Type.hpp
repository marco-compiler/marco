#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <type_traits>

namespace modelica
{
	enum class BuiltinType
	{
		None,
		Integer,
		Float,
		String,
		Boolean,
		Unknown
	};

	template<typename T>
	[[nodiscard]] constexpr BuiltinType typeToBuiltin()
	{
		if constexpr (std::is_same<T, int>())
			return BuiltinType::Integer;

		if constexpr (std::is_same<T, long>())
			return BuiltinType::Integer;

		if constexpr (std::is_same<T, std::string>())
			return BuiltinType::String;

		if constexpr (std::is_same<T, bool>())
			return BuiltinType::Boolean;

		if constexpr (std::is_same<T, float>())
			return BuiltinType::Float;

		if constexpr (std::is_same<T, void>())
			return BuiltinType::None;

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

	class Type
	{
		public:
		Type(BuiltinType type, llvm::ArrayRef<size_t> dim = { 1 });

		template<typename T>
		Type(llvm::ArrayRef<size_t> dim = { 1 })
				: dimensions(llvm::iterator_range<llvm::ArrayRef<size_t>::iterator>(
							std::move(dim))),
					type(typeToBuiltin<T>())
		{
			assert(!dimensions.empty());
		}

		[[nodiscard]] bool operator==(const Type& other) const;
		[[nodiscard]] bool operator!=(const Type& other) const;
		[[nodiscard]] size_t& operator[](int index);
		[[nodiscard]] size_t operator[](int index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] llvm::SmallVectorImpl<size_t>& getDimensions();
		[[nodiscard]] const llvm::SmallVectorImpl<size_t>& getDimensions() const;

		[[nodiscard]] size_t dimensionsCount() const;
		[[nodiscard]] size_t size() const;

		[[nodiscard]] bool isScalar() const;

		[[nodiscard]] llvm::SmallVectorImpl<size_t>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<size_t>::const_iterator begin() const;

		[[nodiscard]] llvm::SmallVectorImpl<size_t>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<size_t>::const_iterator end() const;

		[[nodiscard]] BuiltinType getBuiltIn() const;

		[[nodiscard]] Type subscript(size_t times) const;

		[[nodiscard]] static Type Int();
		[[nodiscard]] static Type Float();
		[[nodiscard]] static Type unknown();

		private:
		llvm::SmallVector<size_t, 3> dimensions;
		BuiltinType type;
	};

	template<typename T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		if constexpr (sizeof...(Args) == 0)
			return Type(typeToBuiltin<T>());

		return Type(typeToBuiltin<T>(), { static_cast<size_t>(args)... });
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
