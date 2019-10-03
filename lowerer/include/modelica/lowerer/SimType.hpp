#pragma once
#include <numeric>

#include "llvm/ADT/SmallVector.h"

namespace modelica
{
	enum class BultinSimTypes
	{
		BOOL,
		INT,
		FLOAT
	};

	template<typename T>
	constexpr BultinSimTypes typeToBuiltin()
	{
		if constexpr (std::is_same<int, T>())
			return BultinSimTypes::INT;
		else if constexpr (std::is_same<bool, T>())
			return BultinSimTypes::BOOL;
		else if constexpr (std::is_same<float, T>())
			return BultinSimTypes::FLOAT;
		else
			assert(false);
	}

	class SimType
	{
		public:
		template<typename... T>
		SimType(BultinSimTypes t, T... args)
				: builtinSimType(t), dimensions({ args... })
		{
		}

		[[nodiscard]] BultinSimTypes getBuiltin() const { return builtinSimType; }

		[[nodiscard]] size_t getDimensionsCount() const
		{
			return dimensions.size();
		}

		[[nodiscard]] size_t getDimension(size_t index) const
		{
			assert(getDimensionsCount() < index);	// NOLINT
			return dimensions[index];
		}

		bool operator==(const SimType& other) const
		{
			if (builtinSimType != other.builtinSimType)
				return false;

			return dimensions == other.dimensions;
		}
		bool operator!=(const SimType& other) const { return !(*this == other); }

		[[nodiscard]] size_t flatSize() const
		{
			return std::accumulate(
					dimensions.begin(),
					dimensions.end(),
					1,
					[](size_t first, size_t second) { return first * second; });
		}

		private:
		BultinSimTypes builtinSimType;
		llvm::SmallVector<size_t, 3> dimensions;
	};
}	// namespace modelica
