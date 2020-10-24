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

		template<typename T>
		[[nodiscard]] T as() const
		{
			if (isA<int>())
				return static_cast<T>(get<int>());

			if (isA<float>())
				return static_cast<T>(get<float>());

			if (isA<bool>())
				return static_cast<T>(get<bool>());

			if (isA<char>())
				return static_cast<T>(get<char>());

			assert(false && "unreachable");
			return {};
		}

		private:
		std::variant<int, float, std::string, char, bool> content;
	};
}	 // namespace modelica
