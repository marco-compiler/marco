#pragma once
#include <numeric>

#include "llvm/ADT/SmallVector.h"

namespace modelica
{
	/**
	 * The simulation can only accepts this 3 foundamental types.
	 * Arrays are built from those too. This types directly map
	 * into llvm int32, int1 and float
	 */
	enum class BultinSimTypes
	{
		BOOL,
		INT,
		FLOAT
	};

	/**
	 * Maps int to BultinSimTypes::BOOL,
	 * bool to BultinSimTypes::INT,
	 * Float to BultinSimTypes::FLOAT
	 *
	 * This is all done at compile time, so it has
	 * no runtime cost
	 */
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
			assert(false && "Unreachable");	// NOLINT
	}

	/**
	 * A simulation type is used to rappresent all
	 * the legal types of simulation variables and simulation expressions.
	 *
	 * SimType is regular types, and follow the rule of 5. They can be moved
	 * but they are larger than 128 bits and are not too cheap to copy or move.
	 * So it's still best to use references when possible.
	 */
	class SimType
	{
		public:
		/**
		 * Overload that allows to pass an arbitrary number of integer to specify
		 * the dimensions of a vector.
		 */
		template<typename... T>
		SimType(BultinSimTypes t, T... args)
				: builtinSimType(t), dimensions({ args... })
		{
		}

		/**
		 * Returns the bultin type.
		 */
		[[nodiscard]] BultinSimTypes getBuiltin() const { return builtinSimType; }

		/**
		 * Return the number of dimensions of this vector.
		 *
		 * Example: int x[10][10] has two dimensions.
		 */
		[[nodiscard]] size_t getDimensionsCount() const
		{
			return dimensions.size();
		}

		/**
		 * Return the size of a particular dimension
		 *
		 * Predocondition: index < getDimensionCount()
		 *
		 * Example: int x[10][5] x.getDimension(1) = 5
		 *
		 */
		[[nodiscard]] size_t getDimension(size_t index) const
		{
			assert(getDimensionsCount() > index);	// NOLINT
			return dimensions[index];
		}

		/**
		 * Deep check for equality. Two types are equal iff
		 * the builtin type is equal and every dimension is equal and
		 * the number of dimensions is the same
		 */
		bool operator==(const SimType& other) const
		{
			if (builtinSimType != other.builtinSimType)
				return false;

			return dimensions == other.dimensions;
		}

		/**
		 * Inverse of operator ==
		 */
		bool operator!=(const SimType& other) const { return !(*this == other); }

		/**
		 * Return the total element count of the array
		 *
		 * Example: int x[10][20] x.flatSize() = 200
		 */
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
