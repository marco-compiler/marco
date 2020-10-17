#pragma once
#include <initializer_list>
#include <numeric>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace modelica
{
	/**
	 * The simulation can only accepts this 3 foundamental types.
	 * Arrays are built from those too. This types directly map
	 * into llvm int32, int1 and float
	 */
	enum class BultinModTypes
	{
		BOOL,
		INT,
		FLOAT
	};

	/**
	 * Maps int to BultinModTypes::BOOL,
	 * bool to BultinModTypes::INT,
	 * Float to BultinModTypes::FLOAT
	 *
	 * This is all done at compile time, so it has
	 * no runtime cost
	 */
	template<typename T>
	constexpr BultinModTypes typeToBuiltin()
	{
		static_assert(
				std::is_same_v<int, T> or std::is_same_v<T, bool> or
						std::is_same_v<T, double>,
				"Unrechable");

		if constexpr (std::is_same<int, T>::value)
			return BultinModTypes::INT;
		else if constexpr (std::is_same<bool, T>::value)
			return BultinModTypes::BOOL;
		else if constexpr (std::is_same<double, T>::value)
			return BultinModTypes::FLOAT;
	}

	/**
	 * A simulation type is used to rappresent all
	 * the legal types of simulation variables and simulation expressions.
	 *
	 * ModType is regular types, and follow the rule of 5. They can be moved
	 * but they are larger than 128 bits and are not too cheap to copy or move.
	 * So it's still best to use references when possible.
	 */
	class ModType
	{
		public:
		explicit ModType(BultinModTypes t): builtinModType(t), dimensions({ 1 }) {}

		ModType(BultinModTypes builtin, llvm::SmallVector<size_t, 3> dim)
				: builtinModType(builtin), dimensions(std::move(dim))
		{
		}

		/**
		 * Overload that allows to pass an arbitrary number of integer to specify
		 * the dimensions of a vector.
		 */
		template<typename First, typename... T>
		ModType(BultinModTypes t, First first, T... args)
				: builtinModType(t),
					dimensions(
							{ static_cast<size_t>(first), static_cast<size_t>(args)... })
		{
		}

		ModType(BultinModTypes builtin, std::initializer_list<size_t> dims)
				: builtinModType(builtin), dimensions(move(dims))
		{
			assert(dimensions.size() != 0);
		}

		/**
		 * \return the bultin type.
		 */
		[[nodiscard]] BultinModTypes getBuiltin() const { return builtinModType; }

		/**
		 * \return the number of dimensions of this vector.
		 *
		 * Example: int x[10][10] has two dimensions.
		 */
		[[nodiscard]] size_t getDimensionsCount() const
		{
			return dimensions.size();
		}

		/**
		 *\return a new type with the same dimensions but different builtin type
		 */
		[[nodiscard]] ModType as(BultinModTypes newBuiltint) const
		{
			return ModType(newBuiltint, dimensions);
		}

		/**
		 * \return the size of a particular dimension
		 *
		 * \require index < getDimensionCount()
		 *
		 * Example: int x[10][5] x.getDimension(1) = 5
		 *
		 */
		[[nodiscard]] size_t getDimension(size_t index) const
		{
			assert(getDimensionsCount() > index);	 // NOLINT
			return dimensions[index];
		}

		/**
		 * \return Deep check for equality. Two types are equal iff
		 * the builtin type is equal and every dimension is equal and
		 * the number of dimensions is the same
		 */
		bool operator==(const ModType& other) const
		{
			if (builtinModType != other.builtinModType)
				return false;

			return dimensions == other.dimensions;
		}

		[[nodiscard]] bool isScalar() const
		{
			return dimensions.size() == 1 && dimensions[0] == 1;
		}

		[[nodiscard]] ModType sclidedType() const
		{
			assert(!isScalar());	// NOLINT
			if (dimensions.size() == 1)
				return ModType(getBuiltin());

			llvm::SmallVector<size_t, 3> dim;
			for (auto i = std::next(std::begin(dimensions));
					 i != std::end(dimensions);
					 i++)
				dim.push_back(*i);

			return ModType(getBuiltin(), std::move(dim));
		}

		/**
		 * \return Inverse of operator ==
		 */
		bool operator!=(const ModType& other) const { return !(*this == other); }

		/**
		 * \return the total element count of the array
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

		/**
		 * A type is said to be castable into another type if the dimensionalites
		 * are matching. \return true if the two types can be casted to each other.
		 */
		[[nodiscard]] bool canBeCastedInto(const ModType& other) const
		{
			return dimensions == other.dimensions;
		}

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		void dumpCSyntax(
				llvm::StringRef name,
				bool useDoubles = true,
				llvm::raw_ostream& OS = llvm::outs()) const;

		/**
		 * \return the array containing the dimensions of this objects
		 */
		[[nodiscard]] const llvm::SmallVector<size_t, 3>& getDimensions() const
		{
			return dimensions;
		}

		private:
		BultinModTypes builtinModType;
		llvm::SmallVector<size_t, 3> dimensions;
	};
}	 // namespace modelica
