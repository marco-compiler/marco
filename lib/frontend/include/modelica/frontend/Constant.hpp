#pragma once

#include <cassert>
#include <string>
#include <variant>
namespace modelica
{
	class Constant
	{
		public:
		explicit Constant(int val): content(val) {}
		explicit Constant(char val): content(val) {}
		explicit Constant(float val): content(val) {}
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

		[[nodiscard]] bool operator==(const Constant& other) const
		{
			return content == other.content;
		}

		[[nodiscard]] bool operator!=(const Constant& other) const
		{
			return !(*this == other);
		}

		private:
		std::variant<int, float, std::string, char> content;
	};
}	 // namespace modelica
