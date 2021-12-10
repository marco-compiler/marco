#include <marco/matching/MCIM.h>
#include <numeric>

using namespace marco::matching;
using namespace marco::matching::detail;

MCIM::IndexesIterator::IndexesIterator(
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

bool MCIM::IndexesIterator::operator==(const MCIM::IndexesIterator& it) const
{
  return eqCurrentIt == it.eqCurrentIt && eqEndIt == it.eqEndIt && varBeginIt == it.varBeginIt && varCurrentIt == it.varCurrentIt && varEndIt == it.varEndIt;
}

bool MCIM::IndexesIterator::operator!=(const MCIM::IndexesIterator& it) const
{
  return eqCurrentIt != it.eqCurrentIt || eqEndIt != it.eqEndIt || varBeginIt != it.varBeginIt || varCurrentIt != it.varCurrentIt || varEndIt != it.varEndIt;
}

MCIM::IndexesIterator& MCIM::IndexesIterator::operator++()
{
  advance();
  return *this;
}

MCIM::IndexesIterator MCIM::IndexesIterator::operator++(int)
{
  auto temp = *this;
  advance();
  return temp;
}

llvm::ArrayRef<long> MCIM::IndexesIterator::operator*() const
{
  return indexes;
}

void MCIM::IndexesIterator::advance()
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

namespace marco::matching::detail
{
  class MCIM::Impl
  {
    public:
    Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    virtual std::unique_ptr<MCIM::Impl> clone() = 0;

    const MultidimensionalRange& getEquationRanges() const;
    const MultidimensionalRange& getVariableRanges() const;

    virtual void apply(const AccessFunction& access) = 0;
    virtual bool get(llvm::ArrayRef<long> indexes) const = 0;
    virtual void set(llvm::ArrayRef<long> indexes) = 0;

    virtual bool empty() const = 0;
    virtual void clear() = 0;

    virtual MCIS flattenEquations() const = 0;
    virtual MCIS flattenVariables() const = 0;

    virtual std::unique_ptr<MCIM::Impl> filterEquations(const MCIS& filter) const = 0;
    virtual std::unique_ptr<MCIM::Impl> filterVariables(const MCIS& filter) const = 0;

    virtual std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const = 0;

    private:
    MultidimensionalRange equationRanges;
    MultidimensionalRange variableRanges;
  };
}

MCIM::Impl::Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
    : equationRanges(std::move(equationRanges)), variableRanges(std::move(variableRanges))
{
}

const MultidimensionalRange& MCIM::Impl::getEquationRanges() const
{
  return equationRanges;
}

const MultidimensionalRange& MCIM::Impl::getVariableRanges() const
{
  return variableRanges;
}

class RegularMCIM : public MCIM::Impl
{
  public:
  class Delta
  {
    public:
    Delta(llvm::ArrayRef<long> keys, llvm::ArrayRef<long> values);

    bool operator==(const Delta& other) const;
    long operator[](size_t index) const;

    size_t size() const;

    Delta inverse() const;

    private:
    llvm::SmallVector<long, 3> values;
  };

  class MCIMElement
  {
    public:
    MCIMElement(MCIS keys, Delta delta);

    const MCIS& getKeys() const;
    void addKeys(MCIS newKeys);
    const Delta& getDelta() const;
    MCIS getValues() const;

    MCIMElement inverse() const;

    private:
    MCIS keys;
    Delta delta;
  };

  RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

  virtual std::unique_ptr<MCIM::Impl> clone() override;

  void apply(const AccessFunction& access) override;
  bool get(llvm::ArrayRef<long> indexes) const override;
  void set(llvm::ArrayRef<long> indexes) override;

  bool empty() const override;
  void clear() override;

  MCIS flattenEquations() const override;
  MCIS flattenVariables() const override;

  std::unique_ptr<MCIM::Impl> filterEquations(const MCIS& filter) const override;
  std::unique_ptr<MCIM::Impl> filterVariables(const MCIS& filter) const override;

  std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const override;

  private:
  void set(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes);
  void add(MCIS keys, Delta delta);

  llvm::SmallVector<MCIMElement, 3> groups;
};

RegularMCIM::Delta::Delta(llvm::ArrayRef<long> keys, llvm::ArrayRef<long> values)
{
  assert(keys.size() == values.size());

  for (const auto& [key, value] : llvm::zip(keys, values))
    this->values.push_back(value - key);
}

bool RegularMCIM::Delta::operator==(const Delta &other) const
{
  return llvm::all_of(llvm::zip(values, other.values), [](const auto& pair) {
      return std::get<0>(pair) == std::get<1>(pair);
  });
}

long RegularMCIM::Delta::operator[](size_t index) const
{
  assert(index < values.size());
  return values[index];
}

size_t RegularMCIM::Delta::size() const
{
  return values.size();
}

RegularMCIM::Delta RegularMCIM::Delta::inverse() const
{
  Delta result(*this);

  for (auto& value : result.values)
    value *= -1;

  return result;
}

RegularMCIM::MCIMElement::MCIMElement(MCIS keys, Delta delta)
        : keys(std::move(keys)), delta(std::move(delta))
{
}

const MCIS& RegularMCIM::MCIMElement::getKeys() const
{
  return keys;
}

void RegularMCIM::MCIMElement::addKeys(MCIS newKeys)
{
  keys.add(std::move(newKeys));
}

const RegularMCIM::Delta& RegularMCIM::MCIMElement::getDelta() const
{
  return delta;
}

MCIS RegularMCIM::MCIMElement::getValues() const
{
  MCIS result;

  for (const auto& range : keys)
  {
    for (const auto& keyRange : keys)
    {
      llvm::SmallVector<Range, 3> valueRanges;

      for (size_t i = 0, e = keyRange.rank(); i < e; ++i)
        valueRanges.emplace_back(keyRange[i].getBegin() + delta[i], keyRange[i].getEnd() + delta[i]);

      result.add(MultidimensionalRange(valueRanges));
    }
  }

  return result;
}

RegularMCIM::MCIMElement RegularMCIM::MCIMElement::inverse() const
{
  return RegularMCIM::MCIMElement(getValues(), delta.inverse());
}

RegularMCIM::RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() == getVariableRanges().rank());
}

std::unique_ptr<MCIM::Impl> RegularMCIM::clone()
{
  return std::make_unique<RegularMCIM>(*this);
}

void RegularMCIM::apply(const AccessFunction& access)
{
  for (const auto& equationIndexes : getEquationRanges())
  {
    assert(access.size() == getVariableRanges().rank());

    llvm::SmallVector<long, 3> variableIndexes;
    access.map(variableIndexes, variableIndexes);
    set(equationIndexes, variableIndexes);
  }
}

bool RegularMCIM::get(llvm::ArrayRef<long> indexes) const
{
  size_t equationRank = getEquationRanges().rank();
  size_t variableRank = getVariableRanges().rank();
  assert(indexes.size() == equationRank + variableRank);

  llvm::SmallVector<long, 3> equationIndexes(indexes.begin(), indexes.begin() + equationRank);
  llvm::SmallVector<long, 3> variableIndexes(indexes.begin() + equationRank, indexes.end());

  Delta delta(equationIndexes, variableIndexes);

  return llvm::any_of(groups, [&](const MCIMElement& group) -> bool {
    return group.getDelta() == delta && group.getKeys().contains(equationIndexes);
  });
}

void RegularMCIM::set(llvm::ArrayRef<long> indexes)
{
  size_t equationRank = getEquationRanges().rank();
  size_t variableRank = getVariableRanges().rank();
  assert(indexes.size() == equationRank + variableRank);

  llvm::SmallVector<long, 3> equationIndexes(indexes.begin(), indexes.begin() + equationRank);
  llvm::SmallVector<long, 3> variableIndexes(indexes.begin() + equationRank, indexes.end());

  set(equationIndexes, variableIndexes);
}

bool RegularMCIM::empty() const
{
  return groups.empty();
}

void RegularMCIM::clear()
{
  groups.clear();
}

MCIS RegularMCIM::flattenEquations() const
{
  MCIS result;

  for (const auto& group : groups)
    result.add(group.getValues());

  return result;
}

MCIS RegularMCIM::flattenVariables() const
{
  MCIS result;

  for (const auto& group : groups)
    result.add(group.getKeys());

  return result;
}

std::unique_ptr<MCIM::Impl> RegularMCIM::filterEquations(const MCIS& filter) const
{
  auto result = std::make_unique<RegularMCIM>(getEquationRanges(), getVariableRanges());

  for (const MCIMElement& group : groups)
    if (auto& equations = group.getKeys(); equations.overlaps(filter))
      result->add(equations.intersect(filter), group.getDelta());

  return result;
}

std::unique_ptr<MCIM::Impl> RegularMCIM::filterVariables(const MCIS& filter) const
{
  auto result = std::make_unique<RegularMCIM>(getEquationRanges(), getVariableRanges());

  for (const auto& group : groups)
  {
    auto invertedGroup = group.inverse();

    if (auto& variables = invertedGroup.getKeys(); variables.overlaps(filter))
    {
      MCIS filteredVariables = variables.intersect(filter);
      MCIMElement filteredVariableGroup(std::move(filteredVariables), invertedGroup.getDelta());
      MCIMElement filteredEquations = filteredVariableGroup.inverse();
      result->add(std::move(filteredEquations.getKeys()), std::move(filteredEquations.getDelta()));
    }
  }

  return result;
}

std::vector<std::unique_ptr<MCIM::Impl>> RegularMCIM::splitGroups() const
{
  std::vector<std::unique_ptr<MCIM::Impl>> result;

  for (const auto& group : groups)
  {
    auto mcim = std::make_unique<RegularMCIM>(getEquationRanges(), getVariableRanges());
    mcim->groups.push_back(group);
    result.push_back(std::move(mcim));
  }

  return result;
}

void RegularMCIM::set(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes)
{
  Delta delta(equationIndexes, variableIndexes);
  llvm::SmallVector<Range, 3> ranges;

  for (const auto& index : equationIndexes)
    ranges.emplace_back(index, index + 1);

  MCIS keys(MultidimensionalRange(std::move(ranges)));
  add(std::move(keys), std::move(delta));
}

void RegularMCIM::add(MCIS keys, Delta delta)
{
  auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
  });

  if (groupIt == groups.end())
    groups.emplace_back(std::move(keys), std::move(delta));
  else
    groupIt->addKeys(std::move(keys));
}

class FlatMCIM : public MCIM::Impl
{
  public:
  class Delta
  {
    public:
    Delta(size_t key, size_t value);

    bool operator==(const Delta& other) const;

    long getValue() const;
    Delta inverse() const;

    private:
    long value;
  };

  class MCIMElement
  {
    public:
    MCIMElement(MCIS keys, Delta delta);

    const MCIS& getKeys() const;
    void addKeys(MCIS newKeys);
    const Delta& getDelta() const;
    MCIS getValues() const;

    MCIMElement inverse() const;

    private:
    MCIS keys;
    Delta delta;
  };

  FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

  std::unique_ptr<MCIM::Impl> clone() override;

  void apply(const AccessFunction& access) override;
  bool get(llvm::ArrayRef<long> indexes) const override;
  void set(llvm::ArrayRef<long> indexes) override;

  bool empty() const override;
  void clear() override;

  MCIS flattenEquations() const override;
  MCIS flattenVariables() const override;

  std::unique_ptr<MCIM::Impl> filterEquations(const MCIS& filter) const override;
  std::unique_ptr<MCIM::Impl> filterVariables(const MCIS& filter) const override;

  std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const override;

  private:
  size_t getFlatEquationIndex(llvm::ArrayRef<long> equationIndexes) const;
  size_t getFlatVariableIndex(llvm::ArrayRef<long> variableIndexes) const;
  std::pair<size_t, size_t> getFlatIndexes(llvm::ArrayRef<long> indexes) const;

  void set(size_t equationFlatIndex, size_t variableFlatIndex);
  void add(MCIS keys, Delta delta);

  llvm::SmallVector<MCIMElement, 3> groups;

  // Stored for faster lookup
  llvm::SmallVector<size_t, 3> equationDimensions;
  llvm::SmallVector<size_t, 3> variableDimensions;
};

static void convertIndexesToZeroBased(
        llvm::ArrayRef<long> indexes,
        const MultidimensionalRange& base,
        llvm::SmallVectorImpl<size_t>& rescaled)
{
  assert(base.rank() == indexes.size());

  for (size_t i = 0, e = base.rank(); i < e; ++i)
  {
    const auto& monoDimRange = base[i];
    long index = indexes[i] - monoDimRange.getBegin();
    assert(index >= 0 && index < monoDimRange.getEnd() - monoDimRange.getBegin());
    rescaled.push_back(index);
  }
}

static void convertIndexesFromZeroBased(
        llvm::ArrayRef<size_t> indexes,
        const MultidimensionalRange& base,
        llvm::SmallVectorImpl<long>& rescaled)
{
  assert(base.rank() == indexes.size());

  for (size_t i = 0, e = base.rank(); i < e; ++i)
  {
    const auto& monoDimRange = base[i];
    long index = indexes[i] + monoDimRange.getBegin();
    assert(index < monoDimRange.getEnd());
    rescaled.push_back(index);
  }
}

/**
 * Get the index to be used to access a flattened array.
 * If an array is declared as [a][b][c], then the access [i][j][k] corresponds
 * to the access [k + c * (j + b * (i))] of the flattened array of size
 * [a * b * c].
 *
 * @param indexes 		access with respect to the original dimensions
 * @param dimensions 	original array dimensions
 * @return flattened index
 */
static size_t flattenIndexes(llvm::ArrayRef<size_t> indexes, llvm::ArrayRef<size_t> dimensions)
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

/**
 * Convert a flattened index into the ones to be used to access the array
 * in its non-flattened version.
 *
 * @param dimensions  original array dimensions
 * @param index       flattened index
 * @param results     where the non-flattened access indexes are saved
 */
static void unflattenIndex(llvm::ArrayRef<size_t> dimensions, size_t index, llvm::SmallVectorImpl<size_t>& results)
{
  assert(dimensions.size() != 0);

  size_t totalSize = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  assert(index < totalSize && "Flattened index exceeds the flat array size");

  size_t size = 1;

  for (size_t i = 1, e = dimensions.size(); i < e; ++i)
    size *= dimensions[i];

  for (size_t i = 1, e = dimensions.size(); i < e; ++i)
  {
    results.push_back(index / size);
    index %= size;
    size /= dimensions[i];
  }

  results.push_back(index);

  assert(size == 1);
  assert(results.size() == dimensions.size());
}

static MCIS flattenMCIS(const MCIS& value, const MultidimensionalRange& range, llvm::ArrayRef<size_t> dimensions)
{
  MCIS result;

  for (const auto& multiDimRange : value)
  {
    llvm::SmallVector<long, 3> firstItemIndexes;
    llvm::SmallVector<long, 3> lastItemIndexes;

    for (size_t i = 0, e = multiDimRange.rank(); i < e; ++i)
    {
      const auto& monoDimRange = multiDimRange[i];
      firstItemIndexes.push_back(monoDimRange.getBegin());
      lastItemIndexes.push_back(monoDimRange.getEnd() - 1);
    }

    llvm::SmallVector<size_t, 3> firstItemRescaled;
    convertIndexesToZeroBased(firstItemIndexes, range, firstItemRescaled);

    llvm::SmallVector<size_t, 3> lastItemRescaled;
    convertIndexesToZeroBased(lastItemIndexes, range, lastItemRescaled);

    size_t firstItemFlattened = flattenIndexes(firstItemRescaled, dimensions);
    size_t lastItemFlattened = flattenIndexes(lastItemRescaled, dimensions);

    result.add(MultidimensionalRange(Range(firstItemFlattened, lastItemFlattened + 1)));
  }

  return result;
}

static MCIS unflattenMCIS(const MCIS& value, const MultidimensionalRange& range, llvm::ArrayRef<size_t> dimensions)
{
  MCIS result;

  for (const auto& multiDimRange : value)
  {
    assert(multiDimRange.rank() == 1);
    auto& monoDimRange = multiDimRange[0];

    size_t firstItemFlattened = monoDimRange.getBegin();
    size_t lastItemFlattened = monoDimRange.getEnd() - 1;

    for (size_t flattened = firstItemFlattened; flattened <= lastItemFlattened; ++flattened)
    {
      llvm::SmallVector<size_t, 3> rescaled;
      unflattenIndex(dimensions, flattened, rescaled);

      llvm::SmallVector<long, 3> indexes;
      convertIndexesFromZeroBased(rescaled, range, indexes);

      llvm::SmallVector<Range, 3> ranges;

      for (const auto& index : indexes)
        ranges.emplace_back(index, index + 1);

      result.add(MultidimensionalRange(std::move(ranges)));
    }
  }

  return result;
}

FlatMCIM::Delta::Delta(size_t key, size_t value) : value(value - key)
{
}

bool FlatMCIM::Delta::operator==(const Delta& other) const
{
  return value == other.value;
}

long FlatMCIM::Delta::getValue() const
{
  return value;
}

FlatMCIM::Delta FlatMCIM::Delta::inverse() const
{
  Delta result(*this);
  result.value *= -1;
  return result;
}

FlatMCIM::MCIMElement::MCIMElement(MCIS keys, Delta delta)
        : keys(std::move(keys)), delta(std::move(delta))
{
}

const MCIS& FlatMCIM::MCIMElement::getKeys() const
{
  return keys;
}

void FlatMCIM::MCIMElement::addKeys(MCIS newKeys)
{
  keys.add(std::move(newKeys));
}

const FlatMCIM::Delta& FlatMCIM::MCIMElement::getDelta() const
{
  return delta;
}

MCIS FlatMCIM::MCIMElement::getValues() const
{
  MCIS result;

  for (const auto& range : keys)
  {
    for (const auto& keyRange : keys)
    {
      llvm::SmallVector<Range, 3> valueRanges;

      for (size_t i = 0, e = keyRange.rank(); i < e; ++i)
        valueRanges.emplace_back(keyRange[i].getBegin() + delta.getValue(), keyRange[i].getEnd() + delta.getValue());

      result.add(MultidimensionalRange(valueRanges));
    }
  }

  return result;
}

FlatMCIM::MCIMElement FlatMCIM::MCIMElement::inverse() const
{
  return FlatMCIM::MCIMElement(getValues(), delta.inverse());
}

FlatMCIM::FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() != getVariableRanges().rank());

  for (size_t i = 0, e = getEquationRanges().rank(); i < e; ++i)
    equationDimensions.push_back(getEquationRanges()[i].size());

  for (size_t i = 0, e = getVariableRanges().rank(); i < e; ++i)
    variableDimensions.push_back(getVariableRanges()[i].size());
}

std::unique_ptr<MCIM::Impl> FlatMCIM::clone()
{
  return std::make_unique<FlatMCIM>(*this);
}

void FlatMCIM::apply(const AccessFunction& access)
{
  for (const auto& equationIndexes : getEquationRanges())
  {
    assert(access.size() == getVariableRanges().rank());

    llvm::SmallVector<long, 3> variableIndexes;
    access.map(variableIndexes, variableIndexes);

    set(getFlatEquationIndex(equationIndexes), getFlatVariableIndex(variableIndexes));
  }
}

bool FlatMCIM::get(llvm::ArrayRef<long> indexes) const
{
  auto flatIndexes = getFlatIndexes(indexes);
  size_t equationFlatIndex = flatIndexes.first;
  size_t variableFlatIndex = flatIndexes.second;

  Delta delta(equationFlatIndex, variableFlatIndex);

  return llvm::any_of(groups, [&](const MCIMElement& group) -> bool {
      return group.getDelta() == delta && group.getKeys().contains(equationFlatIndex);
  });
}

void FlatMCIM::set(llvm::ArrayRef<long> indexes)
{
  auto flatIndexes = getFlatIndexes(indexes);
  set(flatIndexes.first, flatIndexes.second);
}

bool FlatMCIM::empty() const
{
  return groups.empty();
}

void FlatMCIM::clear()
{
  groups.clear();
}

MCIS FlatMCIM::flattenEquations() const
{
  MCIS result;

  for (const auto& group : groups)
    result.add(group.getValues());

  return unflattenMCIS(result, getVariableRanges(), variableDimensions);
}

MCIS FlatMCIM::flattenVariables() const
{
  MCIS result;

  for (const auto& group : groups)
    result.add(group.getKeys());

  return unflattenMCIS(result, getEquationRanges(), equationDimensions);
}

std::unique_ptr<MCIM::Impl> FlatMCIM::filterEquations(const MCIS& filter) const
{
  MCIS flattenedFilter = flattenMCIS(filter, getEquationRanges(), equationDimensions);
  auto result = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());

  for (const MCIMElement& group : groups)
    if (auto& equations = group.getKeys(); equations.overlaps(flattenedFilter))
      result->add(equations.intersect(flattenedFilter), group.getDelta());

  return result;
}

std::unique_ptr<MCIM::Impl> FlatMCIM::filterVariables(const MCIS& filter) const
{
  MCIS flattenedFilter = flattenMCIS(filter, getVariableRanges(), variableDimensions);
  auto result = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());

  for (const auto& group : groups)
  {
    auto invertedGroup = group.inverse();

    if (auto& variables = invertedGroup.getKeys(); variables.overlaps(flattenedFilter))
    {
      MCIS filteredVariables = variables.intersect(flattenedFilter);
      MCIMElement filteredVariableGroup(std::move(filteredVariables), invertedGroup.getDelta());
      MCIMElement filteredEquations = filteredVariableGroup.inverse();
      result->add(std::move(filteredEquations.getKeys()), std::move(filteredEquations.getDelta()));
    }
  }

  return result;
}

std::vector<std::unique_ptr<MCIM::Impl>> FlatMCIM::splitGroups() const
{
  std::vector<std::unique_ptr<MCIM::Impl>> result;

  for (const auto& group : groups)
  {
    auto mcim = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());
    mcim->groups.push_back(group);
    result.push_back(std::move(mcim));
  }

  return result;
}

size_t FlatMCIM::getFlatEquationIndex(llvm::ArrayRef<long> equationIndexes) const
{
  assert(getEquationRanges().rank() == equationIndexes.size());
  llvm::SmallVector<size_t, 3> rescaled;
  convertIndexesToZeroBased(equationIndexes, getEquationRanges(), rescaled);
  return flattenIndexes(rescaled, equationDimensions);
}

size_t FlatMCIM::getFlatVariableIndex(llvm::ArrayRef<long> variableIndexes) const
{
  assert(getVariableRanges().rank() == variableIndexes.size());
  llvm::SmallVector<size_t, 3> rescaled;
  convertIndexesToZeroBased(variableIndexes, getVariableRanges(), rescaled);
  return flattenIndexes(rescaled, variableDimensions);
}

std::pair<size_t, size_t> FlatMCIM::getFlatIndexes(llvm::ArrayRef<long> indexes) const
{
  size_t equationRank = getEquationRanges().rank();
  size_t variableRank = getVariableRanges().rank();
  assert(indexes.size() == equationRank + variableRank);

  llvm::SmallVector<long, 3> equationIndexes(indexes.begin(), indexes.begin() + equationRank);
  llvm::SmallVector<long, 3> variableIndexes(indexes.begin() + equationRank, indexes.end());

  return std::make_pair(getFlatEquationIndex(equationIndexes), getFlatVariableIndex(variableIndexes));
}

void FlatMCIM::set(size_t equationFlatIndex, size_t variableFlatIndex)
{
  Delta delta(equationFlatIndex, variableFlatIndex);
  MCIS keys(MultidimensionalRange(Range(equationFlatIndex, equationFlatIndex + 1)));
  add(std::move(keys), std::move(delta));
}

void FlatMCIM::add(MCIS keys, Delta delta)
{
  auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
  });

  if (groupIt == groups.end())
    groups.emplace_back(std::move(keys), std::move(delta));
  else
    groupIt->addKeys(std::move(keys));
}

MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
{
  if (equationRanges.rank() == variableRanges.rank())
    impl = std::make_unique<RegularMCIM>(std::move(equationRanges), std::move(variableRanges));
  else
    impl = std::make_unique<FlatMCIM>(std::move(equationRanges), std::move(variableRanges));
}

MCIM::MCIM(std::unique_ptr<Impl> impl) : impl(std::move(impl))
{
}

MCIM::MCIM(const MCIM& other) : impl(other.impl->clone())
{
}

MCIM::~MCIM() = default;

const MultidimensionalRange& MCIM::getEquationRanges() const
{
  return impl->getEquationRanges();
}

const MultidimensionalRange& MCIM::getVariableRanges() const
{
  return impl->getVariableRanges();
}

llvm::iterator_range<MCIM::IndexesIterator> MCIM::getIndexes() const
{
  IndexesIterator begin(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.begin();
  });

  IndexesIterator end(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.end();
  });

  return llvm::iterator_range<MCIM::IndexesIterator>(begin, end);
}

void MCIM::apply(const AccessFunction& access)
{
  impl->apply(access);
}

bool MCIM::get(llvm::ArrayRef<long> indexes) const
{
  return impl->get(std::move(indexes));
}

void MCIM::set(llvm::ArrayRef<long> indexes)
{
  impl->set(std::move(indexes));
}

bool MCIM::empty() const
{
  return impl->empty();
}

void MCIM::clear()
{
  impl->clear();
}

MCIS MCIM::flattenEquations() const
{
  return impl->flattenEquations();
}

MCIS MCIM::flattenVariables() const
{
  return impl->flattenVariables();
}

MCIM MCIM::filterEquations(const MCIS& filter) const
{
  return MCIM(impl->filterEquations(filter));
}

MCIM MCIM::filterVariables(const MCIS& filter) const
{
  return MCIM(impl->filterVariables(filter));
}

std::vector<MCIM> MCIM::splitGroups() const
{
  std::vector<MCIM> result;
  auto groups = impl->splitGroups();

  for (auto& group : groups)
    result.push_back(MCIM(std::move(group)));

  return result;
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

static void printIndexes(std::ostream& stream, llvm::ArrayRef<long> indexes)
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
  std::ostream& operator<<(std::ostream& stream, const MCIM& mcim)
  {
    const auto& equationRanges = mcim.getEquationRanges();
    const auto& variableRanges = mcim.getVariableRanges();

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

        stream << (mcim.get(indexes) ? 1 : 0);

        for (size_t i = 0; i < spacesAfter; ++i)
          stream << " ";
      }

      stream << "\n";
    }

    return stream;
  }
}
