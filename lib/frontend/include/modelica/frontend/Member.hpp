#pragma once

#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Type.hpp>
#include <optional>
#include <string>

namespace modelica
{
	class Member
	{
		public:
		Member(
				std::string name,
				Type tp,
				Expression initializer,
				bool isParameter = false)
				: name(std::move(name)),
					type(std::move(tp)),
					initializer(std::move(initializer)),
					isParam(isParameter)
		{
		}

		Member(std::string name, Type tp, bool isParameter = false)
				: name(std::move(name)),
					type(std::move(tp)),
					initializer(std::nullopt),
					isParam(isParameter)
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

		private:
		std::string name;
		Type type;
		std::optional<Expression> initializer;
		bool isParam;
	};

}	 // namespace modelica
