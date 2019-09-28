#pragma once
#include "llvm/ADT/SmallVector.h"

namespace modelica
{
	enum class BultinTypes
	{
		BOOL,
		INT,
		FLOAT
	};

	template<typename T>
	constexpr BultinTypes typeToBuiltin()
	{
		if constexpr (std::is_same<int, T>())
			return BultinTypes::INT;
		else if constexpr (std::is_same<bool, T>())
			return BultinTypes::BOOL;
		else if constexpr (std::is_same<float, T>())
			return BultinTypes::FLOAT;
		else
			assert(false);
	}

	class Type
	{
		public:
		template<typename... T>
		Type(BultinTypes t, T... args): builtinType(t), dimensions({ args... })
		{
		}
		template<BultinTypes t, typename... T>
		Type(T... args): builtinType(t), dimensions({ args... })
		{
		}

		[[nodiscard]] BultinTypes getBuiltin() const { return builtinType; }

		[[nodiscard]] size_t getDimensionsCount() const
		{
			return dimensions.size();
		}

		[[nodiscard]] size_t getDimension(size_t index) const
		{
			assert(getDimensionsCount() < index);	// NOLINT
			return dimensions[index];
		}

		bool operator==(const Type& other) const
		{
			if (builtinType != other.builtinType)
				return false;

			return dimensions == other.dimensions;
		}
		bool operator!=(const Type& other) const { return !(*this == other); }

		private:
		BultinTypes builtinType;
		llvm::SmallVector<size_t, 3> dimensions;
	};
}	// namespace modelica
