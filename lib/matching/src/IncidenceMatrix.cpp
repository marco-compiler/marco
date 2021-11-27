#include <llvm/ADT/STLExtras.h>
#include <marco/matching/IncidenceMatrix.h>
#include <numeric>

using namespace marco::matching;
using namespace marco::matching::detail;

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

IncidenceMatrix::IndexesIterator::IndexesIterator(
        const MultidimensionalRange& equationRange,
        const MultidimensionalRange& variableRange,
        std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction)
    : eqRank(equationRange.rank()),
      eqCurrentIt(initFunction(equationRange)),
      eqEndIt(equationRange.end()),
      varBeginIt(variableRange.begin()),
      varCurrentIt(initFunction(variableRange)),
      varEndIt(variableRange.end()),
      indexes(equationRange.rank() + variableRange.rank(), 0)
{
  if (eqCurrentIt != eqEndIt)
  {
    assert(varCurrentIt != varEndIt);

    for (const auto& index : llvm::enumerate(*eqCurrentIt))
      indexes[index.index()] = index.value();

    for (const auto& index : llvm::enumerate(*varCurrentIt))
      indexes[eqRank + index.index()] = index.value();
  }
}

bool IncidenceMatrix::IndexesIterator::operator==(const IndexesIterator& it) const
{
  return eqCurrentIt == it.eqCurrentIt && eqEndIt == it.eqEndIt && varBeginIt == it.varBeginIt && varCurrentIt == it.varCurrentIt && varEndIt == it.varEndIt;
}

bool IncidenceMatrix::IndexesIterator::operator!=(const IndexesIterator& it) const
{
  return eqCurrentIt != it.eqCurrentIt || eqEndIt != it.eqEndIt || varBeginIt != it.varBeginIt || varCurrentIt != it.varCurrentIt || varEndIt != it.varEndIt;
}

IncidenceMatrix::IndexesIterator& IncidenceMatrix::IndexesIterator::operator++()
{
  advance();
  return *this;
}

IncidenceMatrix::IndexesIterator IncidenceMatrix::IndexesIterator::operator++(int)
{
  auto temp = *this;
  advance();
  return temp;
}

llvm::ArrayRef<long> IncidenceMatrix::IndexesIterator::operator*() const
{
  return indexes;
}

void IncidenceMatrix::IndexesIterator::advance()
{
  if (eqCurrentIt == eqEndIt)
    return;

  ++varCurrentIt;

  if (varCurrentIt == varEndIt)
  {
    ++eqCurrentIt;

    if (eqCurrentIt == eqEndIt)
      return;

    for (const auto& index : llvm::enumerate(*eqCurrentIt))
      indexes[index.index()] = index.value();

    varCurrentIt = varBeginIt;
  }

  for (const auto& index : llvm::enumerate(*varCurrentIt))
    indexes[eqRank + index.index()] = index.value();
}

IncidenceMatrix::IncidenceMatrix(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
		: equationRanges(equationRanges),
			variableRanges(variableRanges),
			data(equationRanges.flatSize(), variableRanges.flatSize(), false)
{
}

void IncidenceMatrix::dump() const
{
  llvm::errs() << *this;
}

IncidenceMatrix IncidenceMatrix::row(MultidimensionalRange variableRanges)
{
  MultidimensionalRange equationRanges(Range(0, 1));

  return IncidenceMatrix(
          std::move(equationRanges),
          std::move(variableRanges));
}

IncidenceMatrix IncidenceMatrix::column(MultidimensionalRange equationRanges)
{
  MultidimensionalRange variableRanges(Range(0, 1));

  return IncidenceMatrix(
          std::move(equationRanges),
          std::move(variableRanges));
}

bool IncidenceMatrix::operator==(const IncidenceMatrix& other) const
{
	if (equationRanges != other.equationRanges)
		return false;

	if (variableRanges != other.variableRanges)
		return false;

	llvm::SmallVector<long, 3> indexes(equationRanges.rank() + variableRanges.rank(), 0);

	for (const auto& equationIndexes : equationRanges)
	{
		for (const auto& equationIndex : llvm::enumerate(equationIndexes))
			indexes[equationIndex.index()] = equationIndex.value();

		for (const auto& variableIndexes : variableRanges)
		{
			for (const auto& variableIndex : llvm::enumerate(variableIndexes))
				indexes[equationRanges.rank() + variableIndex.index()] = variableIndex.value();

			if (get(indexes) != other.get(indexes))
				return false;
		}
	}

	return true;
}

bool IncidenceMatrix::operator!=(const IncidenceMatrix& other) const
{
	if (equationRanges == other.equationRanges)
		return false;

	if (variableRanges == other.variableRanges)
		return false;

	llvm::SmallVector<long, 3> indexes(equationRanges.rank() + variableRanges.rank(), 0);

	for (const auto& equationIndexes : equationRanges)
	{
		for (const auto& equationIndex : llvm::enumerate(equationIndexes))
			indexes[equationIndex.index()] = equationIndex.value();

		for (const auto& variableIndexes : variableRanges)
		{
			for (const auto& variableIndex : llvm::enumerate(variableIndexes))
				indexes[equationRanges.rank() + variableIndex.index()] = variableIndex.value();

			if (get(indexes) != other.get(indexes))
				return true;
		}
	}

	return false;
}

IncidenceMatrix IncidenceMatrix::operator!() const
{
  IncidenceMatrix result = *this;

  for (const auto& indexes : result.getIndexes())
  {
    if (result.get(indexes))
      result.unset(indexes);
    else
      result.set(indexes);
  }

  return result;
}

IncidenceMatrix& IncidenceMatrix::operator+=(const IncidenceMatrix& rhs)
{
	assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
	assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

  for (const auto& indexes : getIndexes())
    if (rhs.get(indexes))
      set(indexes);

	return *this;
}

IncidenceMatrix IncidenceMatrix::operator+(const IncidenceMatrix& rhs) const
{
  IncidenceMatrix result = *this;
  result += rhs;
  return result;
}

IncidenceMatrix& IncidenceMatrix::operator-=(const IncidenceMatrix& rhs)
{
  assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
  assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

  for (const auto& indexes : getIndexes())
  {
    if (get(indexes) && rhs.get(indexes))
      unset(indexes);
  }

  return *this;
}

IncidenceMatrix IncidenceMatrix::operator-(const IncidenceMatrix& rhs) const
{
  IncidenceMatrix result = *this;
  result -= rhs;
  return result;
}

const MultidimensionalRange& IncidenceMatrix::getEquationRanges() const
{
	return equationRanges;
}

const MultidimensionalRange& IncidenceMatrix::getVariableRanges() const
{
	return variableRanges;
}

llvm::iterator_range<IncidenceMatrix::IndexesIterator> IncidenceMatrix::getIndexes() const
{
  IndexesIterator begin(equationRanges, variableRanges, [](const MultidimensionalRange& range) {
    return range.begin();
  });

  IndexesIterator end(equationRanges, variableRanges, [](const MultidimensionalRange& range) {
      return range.end();
  });

  return llvm::iterator_range<IncidenceMatrix::IndexesIterator>(begin, end);
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

size_t IncidenceMatrix::size() const
{
  return llvm::count_if(getIndexes(), [&](const auto& indexes) {
    return get(indexes);
  });
}

void IncidenceMatrix::clear()
{
	data.clear();
}

bool IncidenceMatrix::isEmpty() const
{
  return llvm::none_of(getIndexes(), [&](const auto& indexes) {
      return get(indexes);
  });
}

IncidenceMatrix IncidenceMatrix::flattenEquations() const
{
  IncidenceMatrix result = row(variableRanges);

	llvm::SmallVector<long, 4> sourceIndexes(equationRanges.rank() + variableRanges.rank(), 0);
	llvm::SmallVector<long, 4> destinationIndexes(1 + variableRanges.rank(), 0);

  for (const auto& variableIndexes : variableRanges)
  {
    for (const auto& index : llvm::enumerate(variableIndexes))
    {
      sourceIndexes[equationRanges.rank() + index.index()] = index.value();
      destinationIndexes[1 + index.index()] = index.value();
    }

    for (const auto& equationIndexes : equationRanges)
    {
      for (const auto& index : llvm::enumerate(equationIndexes))
        sourceIndexes[index.index()] = index.value();

      if (get(sourceIndexes))
      {
        result.set(destinationIndexes);
        break;
      }
    }
  }

	return result;
}

IncidenceMatrix IncidenceMatrix::flattenVariables() const
{
  IncidenceMatrix result = column(equationRanges);

  llvm::SmallVector<long, 4> sourceIndexes(equationRanges.rank() + variableRanges.rank(), 0);
  llvm::SmallVector<long, 4> destinationIndexes(equationRanges.rank() + 1, 0);

  for (const auto& equationIndexes : equationRanges)
  {
    for (const auto& index : llvm::enumerate(equationIndexes))
    {
      sourceIndexes[index.index()] = index.value();
      destinationIndexes[index.index()] = index.value();
    }

    for (const auto& variableIndexes : variableRanges)
    {
      for (const auto& index : llvm::enumerate(variableIndexes))
        sourceIndexes[equationRanges.rank() + index.index()] = index.value();

      if (get(sourceIndexes))
      {
        result.set(destinationIndexes);
        break;
      }
    }
  }

	return result;
}

IncidenceMatrix IncidenceMatrix::filterEquations(const IncidenceMatrix& filter) const
{
  assert(filter.getEquationRanges() == equationRanges && "Filter has different equation ranges");
  assert(filter.getVariableRanges().flatSize() == 1 && "Filter must be a column vector");

  IncidenceMatrix result(equationRanges, variableRanges);
  llvm::SmallVector<long, 3> indexes(equationRanges.rank() + variableRanges.rank(), 0);

  for (const auto& filterIndexes : filter.getIndexes())
  {
    for (size_t i = 0; i < equationRanges.rank(); ++i)
      indexes[i] = filterIndexes[i];

    if (filter.get(filterIndexes))
    {
      for (const auto& variableIndexes : variableRanges)
      {
        for (const auto& index : llvm::enumerate(variableIndexes))
          indexes[equationRanges.rank() + index.index()] = index.value();

        if (get(indexes))
          result.set(indexes);
      }
    }
  }

  return result;
}

IncidenceMatrix IncidenceMatrix::filterVariables(const IncidenceMatrix& filter) const
{
  assert(filter.getVariableRanges() == variableRanges && "Filter has different variable ranges");
  assert(filter.getEquationRanges().flatSize() == 1 && "Filter must be a row vector");

  IncidenceMatrix result(equationRanges, variableRanges);
  llvm::SmallVector<long, 3> indexes(equationRanges.rank() + variableRanges.rank(), 0);

  for (const auto& filterIndexes : filter.getIndexes())
  {
    for (size_t i = 0; i < filter.getVariableRanges().rank(); ++i)
      indexes[equationRanges.rank() + i] = filterIndexes[filter.getEquationRanges().rank() + i];

    if (filter.get(filterIndexes))
    {
      for (const auto& equationIndexes : equationRanges)
      {
        for (const auto& index : llvm::enumerate(equationIndexes))
          indexes[index.index()] = index.value();

        if (get(indexes))
          result.set(indexes);
      }
    }
  }

  return result;
}

bool IncidenceMatrix::isDisjoint(const IncidenceMatrix& other) const
{
  assert(equationRanges == other.equationRanges);
  assert(variableRanges == other.variableRanges);

  return llvm::none_of(getIndexes(), [&](auto indexes) {
    return get(indexes) && other.get(indexes);
  });
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
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd() - range.getBegin());
		equationIndexes.push_back(scaledIndex);
	}

	for (size_t i = equationRanges.rank(), e = equationRanges.rank() + variableRanges.rank(); i < e; ++i)
	{
		auto range = variableRanges[i - equationRanges.rank()];
		auto scaledIndex = indexes[i] - range.getBegin();
		assert(scaledIndex >= 0 && scaledIndex < range.getEnd() - range.getBegin());
		variableIndexes.push_back(scaledIndex);
	}

	// Flatten the sets of indexes so that they can be used to uniquely refer to
	// an element within the 2D data structure.

	size_t row = flattenAccess(equationDimensions, equationIndexes);
	size_t column = flattenAccess(variableDimensions, variableIndexes);

	return std::make_pair(row, column);
}

template <class T>
static size_t numDigits(T value)
{
	if (value > -10 && value < 10)
		return 1;

	size_t digits = 0;

	while (value != 0) {
		value /= 10;
		++digits;
	}

	return digits;
}

static size_t getRangeMaxColumns(const Range& range)
{
	size_t beginDigits = numDigits(range.getBegin());
	size_t endDigits = numDigits(range.getEnd());

	if (range.getBegin() < 0)
		++beginDigits;

	if (range.getEnd() < 0)
		++endDigits;

	return std::max(beginDigits, endDigits);
}

static size_t getIndexesWidth(llvm::ArrayRef<long> indexes)
{
	size_t result = 0;

	for (const auto& index : indexes)
	{
		result += numDigits(index);

		if (index < 0)
			++result;
	}

	return result;
}

static size_t getWrappedIndexesLength(size_t indexesLength, size_t numberOfIndexes)
{
	size_t result = indexesLength;

	result += 1; // '(' character
	result += numberOfIndexes - 1; // ',' characters
	result += 1; // ')' character

	return result;
}

static void printIndexes(llvm::raw_ostream& stream, llvm::ArrayRef<long> indexes)
{
	bool separator = false;
	stream << "(";

	for (const auto& index : indexes)
	{
		if (separator)
			stream << ",";

		separator = true;
		stream << index;
	}

	stream << ")";
}

namespace marco::matching::detail
{
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const IncidenceMatrix& matrix)
	{
		const auto& equationRanges = matrix.getEquationRanges();
		const auto& variableRanges = matrix.getVariableRanges();

		// Determine the max widths of the indexes of the equation, so that they
		// will be properly aligned.
		llvm::SmallVector<size_t, 3> equationIndexesCols;

		for (size_t i = 0, e = equationRanges.rank(); i < e; ++i)
			equationIndexesCols.push_back(getRangeMaxColumns(equationRanges[i]));

		size_t equationIndexesMaxWidth = std::accumulate(equationIndexesCols.begin(), equationIndexesCols.end(), 0);
		size_t equationIndexesColumnWidth = getWrappedIndexesLength(equationIndexesMaxWidth, equationRanges.rank());

		// Determine the max column width, so that the horizontal spacing is the
		// same among all the items.
		llvm::SmallVector<size_t, 3> variableIndexesCols;

		for (size_t i = 0, e = variableRanges.rank(); i < e; ++i)
			variableIndexesCols.push_back(getRangeMaxColumns(variableRanges[i]));

		size_t variableIndexesMaxWidth = std::accumulate(variableIndexesCols.begin(), variableIndexesCols.end(), 0);
		size_t variableIndexesColumnWidth = getWrappedIndexesLength(variableIndexesMaxWidth, variableRanges.rank());

		// Print the spacing of the first line
		for (size_t i = 0, e = equationIndexesColumnWidth; i < e; ++i)
			stream << " ";

		// Print the variable indexes
		for (const auto& variableIndexes : variableRanges)
		{
			stream << " ";
			size_t columnWidth = getIndexesWidth(variableIndexes);

			for (size_t i = columnWidth; i < variableIndexesMaxWidth; ++i)
				stream << " ";

			printIndexes(stream, variableIndexes);
		}

		// The first line containing the variable indexes is finished
		stream << "\n";

		// Print a line for each equation
		llvm::SmallVector<long, 4> indexes;

		for (const auto& equationIndexes : equationRanges)
		{
			for (size_t i = getIndexesWidth(equationIndexes); i < equationIndexesMaxWidth; ++i)
				stream << " ";

			printIndexes(stream, equationIndexes);

			for (const auto& variableIndexes : variableRanges)
			{
				stream << " ";

				indexes.clear();
				indexes.insert(indexes.end(), equationIndexes.begin(), equationIndexes.end());
				indexes.insert(indexes.end(), variableIndexes.begin(), variableIndexes.end());

				size_t columnWidth = variableIndexesColumnWidth;
				size_t spacesAfter = (columnWidth - 1) / 2;
				size_t spacesBefore = columnWidth - 1 - spacesAfter;

				for (size_t i = 0; i < spacesBefore; ++i)
					stream << " ";

				stream << (matrix.get(indexes) ? 1 : 0);

				for (size_t i = 0; i < spacesAfter; ++i)
					stream << " ";
			}

			stream << "\n";
		}

		return stream;
	}

	std::ostream& operator<<(std::ostream& stream, const IncidenceMatrix& matrix)
	{
		llvm::raw_os_ostream(stream) << matrix;
		return stream;
	}
}
