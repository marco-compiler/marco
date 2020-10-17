#pragma once

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "llvm/Support/raw_ostream.h"
#include "modelica/frontend/Type.hpp"

namespace modelica
{
	template<typename T>
	BuiltinType typeToFrontendType()
	{
		if constexpr (std::is_same<T, double>::value)
			return BuiltinType::Float;
		if constexpr (std::is_same<T, int>::value)
			return BuiltinType::Integer;
		if constexpr (std::is_same<T, bool>::value)
			return BuiltinType::Boolean;
		if constexpr (std::is_same<std::string, T>::value)
			return BuiltinType::String;
		assert(false && "unreachable");
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

	class Constant
	{
		public:
		explicit Constant(int val): content(val) {}
		explicit Constant(char val): content(val) {}
		explicit Constant(float val): content(static_cast<double>(val)) {}
		explicit Constant(bool val): content(val) {}
		explicit Constant(double val): content(val) {}
		explicit Constant(std::string val): content(std::move(val)) {}

		template<BuiltinType T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<frontendTypeToType_v<T>>(content);
		}

		template<BuiltinType T>
		[[nodiscard]] const frontendTypeToType_v<T>& get() const
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(content);
		}

		template<BuiltinType T>
		[[nodiscard]] frontendTypeToType_v<T>& get()
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(content);
		}

		template<BuiltinType T>
		[[nodiscard]] frontendTypeToType_v<T> as() const
		{
			using Tr = frontendTypeToType_v<T>;
			if (isA<BuiltinType::Integer>())
				return static_cast<Tr>(get<BuiltinType::Integer>());

			if (isA<BuiltinType::Float>())
				return static_cast<Tr>(get<BuiltinType::Float>());

			if (isA<BuiltinType::Boolean>())
				return static_cast<Tr>(get<BuiltinType::Boolean>());

			assert(false && "unrechable");
			return {};
		}

		[[nodiscard]] bool operator==(const Constant& other) const
		{
			return content == other.content;
		}

		[[nodiscard]] bool operator!=(const Constant& other) const
		{
			return !(*this == other);
		}

		void dump(
				llvm::raw_ostream& OS = llvm::outs(), size_t indentLevel = 0) const;

		private:
		std::variant<int, double, std::string, bool> content;
	};
}	 // namespace modelica
