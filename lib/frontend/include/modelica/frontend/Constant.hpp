#pragma once

#include <cassert>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <type_traits>
#include <variant>

#include "Type.hpp"

namespace modelica
{
	class Constant
	{
		public:
		explicit Constant(int val);
		explicit Constant(char val);
		explicit Constant(float val);
		explicit Constant(bool val);
		explicit Constant(double val);
		explicit Constant(std::string val);

		[[nodiscard]] bool operator==(const Constant& other) const;
		[[nodiscard]] bool operator!=(const Constant& other) const;

		void dump() const;
		void dump(
				llvm::raw_ostream& OS = llvm::outs(), size_t indentLevel = 0) const;

		template<BuiltinType T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<frontendTypeToType_v<T>>(content);
		}

		template<BuiltinType T>
		[[nodiscard]] frontendTypeToType_v<T>& get()
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(content);
		}

		template<BuiltinType T>
		[[nodiscard]] const frontendTypeToType_v<T>& get() const
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

			assert(false && "unreachable");
			return {};
		}

		private:
		std::variant<int, double, std::string, bool> content;
	};
}	 // namespace modelica
