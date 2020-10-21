#pragma once

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
		explicit Type(BuiltinType type, llvm::SmallVector<size_t, 3> dim = { 1 })
				: dimensions(std::move(dim)), type(type)
		{
			assert(!dimensions.empty());
		}

		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

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
