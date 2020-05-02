#pragma once

#include <cassert>
#include <string>
#include <variant>

#include "llvm/Support/raw_ostream.h"
namespace modelica
{
	class Constant
	{
		public:
		explicit Constant(int val): content(val) {}
		explicit Constant(char val): content(val) {}
		explicit Constant(float val): content(val) {}
		explicit Constant(bool val): content(val) {}
		explicit Constant(double val): content(static_cast<float>(val)) {}
		explicit Constant(std::string val): content(std::move(val)) {}

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] T& get()
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
				llvm::raw_ostream& OS = llvm::outs(), size_t indentLevel = 0) const
		{
			OS.indent(indentLevel);
			if (isA<int>())
				OS << get<int>();
			else if (isA<float>())
				OS << get<float>();
			else if (isA<bool>())
				OS << (get<bool>() ? "true" : "false");
			else if (isA<char>())
				OS << get<char>();
			else if (isA<std::string>())
				OS << get<std::string>();
		}

		private:
		std::variant<int, float, std::string, char, bool> content;
	};
}	 // namespace modelica
