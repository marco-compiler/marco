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
				std::optional<Constant> startOverload = std::nullopt)
				: name(std::move(name)),
					type(std::move(tp)),
					initializer(std::move(initializer)),
					isParam(isParameter),
					startOverload(std::move(startOverload))
		{
		}

		Member(
				std::string name,
				Type tp,
				bool isParameter = false,
				std::optional<Constant> startOverload = std::nullopt)
				: name(std::move(name)),
					type(std::move(tp)),
					initializer(std::nullopt),
					isParam(isParameter),
					startOverload(std::move(startOverload))
		{
		}

		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] std::string& getName() { return name; }
		[[nodiscard]] const Type& getType() const { return type; }
		[[nodiscard]] Type& getType() { return type; }
		[[nodiscard]] bool hasInitializer() const
		{
			return initializer.has_value();
		}

		[[nodiscard]] const Expression& getInitializer() const
		{
			assert(hasInitializer());
			return *initializer;
		}

		[[nodiscard]] Expression& getInitializer()
		{
			assert(hasInitializer());
			return *initializer;
		}

		[[nodiscard]] bool hasStartOverload() const
		{
			return startOverload.has_value();
		}

		[[nodiscard]] const Constant& getStartOverload() const
		{
			assert(hasStartOverload());
			return startOverload.value();
		}

		[[nodiscard]] Constant& getStartOverload()
		{
			assert(hasStartOverload());
			return startOverload.value();
		}

		[[nodiscard]] bool operator==(const Member& other) const
		{
			return name == other.name && type == other.type &&
						 initializer == other.initializer;
		}

		[[nodiscard]] bool operator!=(const Member& other) const
		{
			return !(*this == other);
		}
		[[nodiscard]] bool isParameter() const { return isParam; }

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0);

		private:
		std::string name;
		Type type;
		std::optional<Expression> initializer;
		bool isParam;

		std::optional<Constant> startOverload;
	};

}	 // namespace modelica
