#pragma once

#include <optional>
#include <string>

#include "Constant.h"
#include "Expression.h"
#include "Type.h"
#include "TypePrefix.h"

namespace modelica
{
	class Member
	{
		public:
		Member(
				SourcePosition location,
				std::string name,
				Type tp,
				TypePrefix prefix,
				Expression initializer,
				bool isPublic = true,
				std::optional<Expression> startOverload = std::nullopt);

		Member(
				SourcePosition location,
				std::string name,
				Type tp,
				TypePrefix typePrefix,
				bool isPublic = true,
				std::optional<Expression> startOverload = std::nullopt);

		[[nodiscard]] bool operator==(const Member& other) const;
		[[nodiscard]] bool operator!=(const Member& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

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

		[[nodiscard]] bool isPublic() const;
		[[nodiscard]] bool isParameter() const;
		[[nodiscard]] bool isInput() const;
		[[nodiscard]] bool isOutput() const;

		private:
		SourcePosition location;
		std::string name;
		Type type;
		TypePrefix typePrefix;
		std::optional<Expression> initializer;
		bool isPublicMember;

		std::optional<Expression> startOverload;
	};

}	 // namespace modelica
