#pragma once

#include <cassert>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>
#include <string>
#include <type_traits>
#include <variant>

#include "Type.h"

namespace modelica
{
	class Constant
	{
		public:
		Constant(SourcePosition location, bool val);
		Constant(SourcePosition location, int val);
		Constant(SourcePosition location, float val);
		Constant(SourcePosition location, double val);
		Constant(SourcePosition location, char val);
		Constant(SourcePosition location, std::string val);

		[[nodiscard]] bool operator==(const Constant& other) const;
		[[nodiscard]] bool operator!=(const Constant& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

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
		SourcePosition location;
		std::variant<bool, int, double, std::string> content;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Constant& obj);

	std::string toString(const Constant& obj);
}	 // namespace modelica
