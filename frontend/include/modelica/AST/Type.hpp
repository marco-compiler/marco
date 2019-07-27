#pragma once

#include <string>
#include <vector>

namespace modelica
{
	enum class BuiltinType
	{
		None,
		Integer,
		Float,
		String,
		Boolean,
		UserDefined,
		Unknown
	};
	class Type
	{
		public:
		Type(
				BuiltinType type,
				std::vector<int> dim = std::vector({ 1 }),
				std::string userDefinedType = "")
				: userDefinedType(std::move(userDefinedType)),
					dimensions(std::move(dim)),
					type(type)
		{
		}

		template<typename... Args>
		Type(
				BuiltinType type, Args... args, const std::string& userDefinedType = "")
				: userDefinedType(std::move(userDefinedType)),
					dimensions({ args... }),
					type(type)
		{
		}

		[[nodiscard]] bool isUserDefined() const
		{
			return type == BuiltinType::UserDefined;
		}
		[[nodiscard]] int size() const { return dimensions.size(); }
		[[nodiscard]] BuiltinType getBuiltIn() const { return type; }
		[[nodiscard]] auto begin() { return dimensions.begin(); }
		[[nodiscard]] auto end() { return dimensions.end(); }
		[[nodiscard]] auto cbegin() const { return dimensions.cbegin(); }
		[[nodiscard]] auto cend() const { return dimensions.cend(); }
		int& operator[](int index) { return dimensions[index]; }
		const int& operator[](int index) const { return dimensions[index]; }
		[[nodiscard]] const std::string& getUserDefinedType() const
		{
			return userDefinedType;
		}

		[[nodiscard]] bool operator==(const Type& other) const
		{
			bool correctDim = type == other.type && dimensions == other.dimensions;
			if (type != BuiltinType::UserDefined)
				return correctDim;

			return correctDim && userDefinedType == other.userDefinedType;
		}
		[[nodiscard]] bool operator!=(const Type& other) const
		{
			return !operator==(other);
		}

		private:
		std::string userDefinedType;
		std::vector<int> dimensions;
		BuiltinType type;
	};
}	// namespace modelica
