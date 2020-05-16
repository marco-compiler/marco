#pragma once

#include <llvm/ADT/SmallVector.h>
#include <string>
#include <type_traits>

#include "llvm/Support/raw_ostream.h"

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

	inline std::string builtinToString(BuiltinType type)
	{
		switch (type)
		{
			case BuiltinType::None:
				return "None";
			case BuiltinType::Integer:
				return "Integer";
			case BuiltinType::Float:
				return "Float";
			case BuiltinType::String:
				return "String";
			case BuiltinType::Boolean:
				return "Boolean";
			case BuiltinType::Unknown:
				return "Unknown";
		}
		assert(false && "unrechable");
	}

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
			assert(!dimensions.empty());
		}

		template<typename T>
		explicit Type(llvm::SmallVector<size_t, 3> dim = { 1 })
				: dimensions(std::move(dim)), type(typeToBuiltin<T>())
		{
			assert(!dimensions.empty());
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
		[[nodiscard]] size_t& operator[](int index) { return dimensions[index]; }
		[[nodiscard]] size_t operator[](int index) const
		{
			return dimensions[index];
		}

		[[nodiscard]] bool operator==(const Type& other) const
		{
			return type == other.type && dimensions == other.dimensions;
		}

		[[nodiscard]] bool operator!=(const Type& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] static Type unkown() { return Type(BuiltinType::Unknown); }

		[[nodiscard]] Type subscript(size_t times) const
		{
			assert(!isScalar());
			if (dimensions.size() == times)
				return Type(type);

			assert(times > dimensions.size());
			return Type(
					type,
					llvm::SmallVector<size_t, 3>(
							dimensions.begin() + times, dimensions.end()));
		}
		[[nodiscard]] auto& getDimensions() { return dimensions; }
		[[nodiscard]] const auto& getDimensions() const { return dimensions; }
		[[nodiscard]] bool isScalar() const
		{
			return dimensions.size() == 1 && dimensions[0] == 1;
		}

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0) const
		{
			OS << builtinToString(type);
			if (!isScalar())
				for (size_t dim : dimensions)
				{
					OS << dim;
					OS << " ";
				}
		}

		static Type Int() { return Type(typeToBuiltin<int>()); }
		static Type Float() { return Type(typeToBuiltin<float>()); }

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
}	 // namespace modelica
