#pragma once

#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Type.hpp>
#include <optional>
#include <string>

#include "modelica/frontend/Constant.hpp"

namespace modelica
{
	class Member
	{
		public:
		Member(
				std::string name,
				Type tp,
				Expression initializer,
				bool isParameter = false,
				std::optional<Expression> startOverload = std::nullopt);

		Member(
				std::string name,
				Type tp,
				bool isParameter = false,
				std::optional<Expression> startOverload = std::nullopt);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] bool operator==(const Member& other) const;
		[[nodiscard]] bool operator!=(const Member& other) const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;

		[[nodiscard]] bool hasInitializer() const;
		[[nodiscard]] Expression& getInitializer();
		[[nodiscard]] const Expression& getInitializer() const;

		[[nodiscard]] bool hasStartOverload() const;
		[[nodiscard]] Expression& getStartOverload();
		[[nodiscard]] const Expression& getStartOverload() const;

		[[nodiscard]] bool isParameter() const;

		private:
		std::string name;
		Type type;
		std::optional<Expression> initializer;
		bool isParam;

		std::optional<Expression> startOverload;
	};

}	 // namespace modelica
