#pragma once

#include <llvm/ADT/SmallVector.h>
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

	class Type
	{
		public:
		explicit Type(BuiltinType type, llvm::SmallVector<size_t, 3> dim = { 1 })
				: dimensions(std::move(dim)), type(type)
		{
		}

		template<typename T>
		explicit Type(llvm::SmallVector<size_t, 3> dim = { 1 })
				: dimensions(std::move(dim)), type(typeToBuiltin<T>())
		{
		}

		[[nodiscard]] size_t dimensionsCount() const { return dimensions.size(); }
		[[nodiscard]] size_t size() const
		{
			size_t toReturn = 1;
			for (size_t dim : dimensions)
				toReturn *= dim;
			return toReturn;
		}
		[[nodiscard]] BuiltinType getBuiltIn() const { return type; }
		[[nodiscard]] auto begin() { return dimensions.begin(); }
		[[nodiscard]] auto end() { return dimensions.end(); }
		[[nodiscard]] auto begin() const { return dimensions.begin(); }
		[[nodiscard]] auto end() const { return dimensions.end(); }
		[[nodiscard]] auto operator[](int index) { return dimensions[index]; }
		[[nodiscard]] auto operator[](int index) const { return dimensions[index]; }

		[[nodiscard]] bool operator==(const Type& other) const
		{
			return type == other.type && dimensions == other.dimensions;
		}

		[[nodiscard]] bool operator!=(const Type& other) const
		{
			return !(*this == other);
		}

		private:
		llvm::SmallVector<size_t, 3> dimensions;
		BuiltinType type;
	};

	template<typename T, typename... Args>
	[[nodiscard]] Type makeType(Args... args)
	{
		return Type(typeToBuiltin<T>(), args...);
	}
}	 // namespace modelica
