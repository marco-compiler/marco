#pragma once

#include <cassert>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Type.hpp>
#include <string>
#include <type_traits>
#include <variant>

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
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		template<class Visitor>
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		template<BuiltInType T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<frontendTypeToType_v<T>>(content);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T>& get()
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(content);
		}

		template<BuiltInType T>
		[[nodiscard]] const frontendTypeToType_v<T>& get() const
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(content);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T> as() const
		{
			using Tr = frontendTypeToType_v<T>;

			if (isA<BuiltInType::Integer>())
				return static_cast<Tr>(get<BuiltInType::Integer>());

			if (isA<BuiltInType::Float>())
				return static_cast<Tr>(get<BuiltInType::Float>());

			if (isA<BuiltInType::Boolean>())
				return static_cast<Tr>(get<BuiltInType::Boolean>());

			assert(false && "unreachable");
			return {};
		}

		private:
		std::variant<int, double, std::string, bool> content;
	};
}	 // namespace modelica
