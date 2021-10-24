#include <marco/matching/IncidenceMatrix.h>

using namespace marco::matching;

/**
 * Get the index to be used to access the flattened array.
 * If an array is declared as [a][b][c], then the access [i][j][k] corresponds
 * to the access [k + c * (j + b * (i))] of the flattened array of size
 * [a * b * c].
 *
 * @param dimensions 	original array dimensions
 * @param indexes 		access with respect to the original dimensions
 * @return flattened array index
 */
static size_t flattenAccess(llvm::ArrayRef<size_t> dimensions, llvm::ArrayRef<size_t> indexes)
{
	assert(dimensions.size() == indexes.size());
	size_t result = 0;

	for (auto index : llvm::enumerate(indexes))
	{
		result += index.value();

		if (index.index() < indexes.size() - 1)
			result *= dimensions[index.index() + 1];
	}

	return result;
}

IncidenceMatrix::IncidenceMatrix(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
		: equationRanges(equationRanges),
			variableRanges(variableRanges),
			data(equationRanges.flatSize(), variableRanges.flatSize(), false)
{
	assert(variableRanges.rank() <= equationRanges.rank());
}

void IncidenceMatrix::apply(AccessFunction access)
{
	llvm::SmallVector<long, 6> indexes;

	for (const auto& equationIndexes : equationRanges)
	{
		assert(equationIndexes.size() == equationRanges.rank());
		assert(access.size() == variableRanges.rank());

		indexes.clear();
		indexes.insert(indexes.begin(), equationIndexes.begin(), equationIndexes.end());

		for (const auto& singleDimensionAccess : access)
			access.map(indexes, equationIndexes);

		assert(indexes.size() == equationRanges.rank() + variableRanges.rank());
		set(indexes);
	}
}

bool IncidenceMatrix::get(llvm::ArrayRef<long> indexes) const
{
	auto matrixIndexes = getMatrixIndexes(indexes);
	return data(matrixIndexes.first, matrixIndexes.second);
}

void IncidenceMatrix::set(llvm::ArrayRef<long> indexes)
{
	auto matrixIndexes = getMatrixIndexes(indexes);
	data(matrixIndexes.first, matrixIndexes.second) = true;
}

void IncidenceMatrix::unset(llvm::ArrayRef<long> indexes)
{
	auto matrixIndexes = getMatrixIndexes(indexes);
	data(matrixIndexes.first, matrixIndexes.second) = false;
}

void IncidenceMatrix::splitIndexes(llvm::ArrayRef<long> indexes, llvm::SmallVectorImpl<size_t>& equationIndexes, llvm::SmallVectorImpl<size_t>& variableIndexes) const
{
	assert(equationRanges.rank() + variableRanges.rank() == indexes.size());

	for (size_t i = 0, e = equationRanges.rank(); i < e; ++i)
	{
		auto range = equationRanges[i];
		auto scaledIndex = indexes[i] - range.getBegin();
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd());
	}

	equationIndexes.insert(equationIndexes.begin(), indexes.begin(), indexes.begin() + equationRanges.rank());
	variableIndexes.insert(variableIndexes.begin(), indexes.begin() + equationRanges.rank(), indexes.end());

	for (size_t i = equationRanges.rank() + 1, e = equationRanges.rank() + variableRanges.rank(); i < e; ++i)
	{
		auto range = equationRanges[i];
		auto scaledIndex = indexes[i] - range.getBegin();
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd());
	}
}

std::pair<size_t, size_t> IncidenceMatrix::getMatrixIndexes(llvm::ArrayRef<long> indexes) const
{
	// Determine the sizes of the equation's dimensions
	llvm::SmallVector<size_t> equationDimensions;
	equationRanges.getSizes(equationDimensions);

	// Determine the sizes of the variable's dimensions
	llvm::SmallVector<size_t> variableDimensions;
	variableRanges.getSizes(variableDimensions);

	// Split the indexes among equation and variable ones, and also rescale them
	// according to each range start, so that the result is 0-based.

	llvm::SmallVector<size_t> equationIndexes;
	llvm::SmallVector<size_t> variableIndexes;

	assert(equationRanges.rank() + variableRanges.rank() == indexes.size());

	for (size_t i = 0, e = equationRanges.rank(); i < e; ++i)
	{
		auto range = equationRanges[i];
		auto scaledIndex = indexes[i] - range.getBegin();
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd());
		equationIndexes.push_back(scaledIndex);
	}

	for (size_t i = equationRanges.rank(), e = equationRanges.rank() + variableRanges.rank(); i < e; ++i)
	{
		auto range = variableRanges[i - equationRanges.rank()];
		auto scaledIndex = indexes[i] - range.getBegin();
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd());
		variableIndexes.push_back(scaledIndex);
	}

	size_t row = flattenAccess(equationDimensions, equationIndexes);
	size_t column = flattenAccess(variableDimensions, variableIndexes);

	return std::make_pair(row, column);
}
