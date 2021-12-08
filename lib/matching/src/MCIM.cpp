#include <marco/matching/MCIM.h>
#include <numeric>

using namespace marco::matching;

namespace marco::matching
{
  class MCIM::Impl
  {
    public:
    Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    const MultidimensionalRange& getEquationRanges() const;
    const MultidimensionalRange& getVariableRanges() const;

    virtual void apply(const AccessFunction& access) = 0;
    virtual bool get(llvm::ArrayRef<long> indexes) const = 0;
    virtual void set(llvm::ArrayRef<long> indexes) = 0;

    virtual bool empty() const = 0;
    virtual void clear() = 0;

    virtual MCIS flattenEquations() const = 0;

    virtual MCIS flattenVariables() const = 0;

    virtual MCIM filterEquations(const MCIS& filter) const = 0;

    virtual MCIM filterVariables(const MCIS& filter) const = 0;

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
    Delta(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes);

    bool operator==(const Delta& other) const;
    long operator[](size_t index) const;

    size_t size() const;

    private:
    llvm::SmallVector<long, 3> values;
  };

  class MCIMElement
  {
    public:
    MCIMElement(MCIS k, Delta delta);

    const MCIS& getEquations() const;
    void addEquation(MultidimensionalRange equationIndexes);
    const Delta& getDelta() const;
    MCIS getVariables() const;

    private:
    MCIS k;
    Delta delta;
  };

  RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

  void apply(const AccessFunction& access) override;
  bool get(llvm::ArrayRef<long> indexes) const override;
  void set(llvm::ArrayRef<long> indexes) override;

  bool empty() const override;
  void clear() override;

  MCIS flattenEquations() const override;

  MCIS flattenVariables() const override;

  MCIM filterEquations(const MCIS& filter) const override;

  MCIM filterVariables(const MCIS& filter) const override;

  private:
  void set(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes);

  llvm::SmallVector<MCIMElement, 3> groups;
};

RegularMCIM::Delta::Delta(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes)
{
  assert(equationIndexes.size() == variableIndexes.size());

  for (const auto& [equationIndex, variableIndex] : llvm::zip(equationIndexes, variableIndexes))
    values.push_back(variableIndex - equationIndex);
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

RegularMCIM::MCIMElement::MCIMElement(MCIS k, Delta delta)
        : k(std::move(k)), delta(std::move(delta))
{
}

const MCIS& RegularMCIM::MCIMElement::getEquations() const
{
  return k;
}

void RegularMCIM::MCIMElement::addEquation(MultidimensionalRange equationIndexes)
{
  k.add(std::move(equationIndexes));
}

const RegularMCIM::Delta& RegularMCIM::MCIMElement::getDelta() const
{
  return delta;
}

MCIS RegularMCIM::MCIMElement::getVariables() const
{
  MCIS result;

  for (const auto& range : k)
  {
    for (const auto& equationRange : k)
    {
      llvm::SmallVector<Range, 3> variableRanges;

      for (size_t i = 0, e = equationRange.rank(); i < e; ++i)
        variableRanges.emplace_back(equationRange[i].getBegin() + delta[i], equationRange[i].getEnd() + delta[i]);

      result.add(MultidimensionalRange(variableRanges));
    }
  }

  return result;
}

RegularMCIM::RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() == getVariableRanges().rank());
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
    return group.getDelta() == delta && group.getEquations().contains(equationIndexes);
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
    result.add(group.getVariables());

  return result;
}

MCIS RegularMCIM::flattenVariables() const
{
  MCIS result;

  for (const auto& group : groups)
    result.add(group.getEquations());

  return result;
}

MCIM RegularMCIM::filterEquations(const MCIS& filter) const
{
  // TODO
}

MCIM RegularMCIM::filterVariables(const MCIS& filter) const
{
  // TODO
}

void RegularMCIM::set(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes)
{
  Delta delta(equationIndexes, variableIndexes);

  auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
  });

  llvm::SmallVector<Range, 3> ranges;

  for (const auto& index : equationIndexes)
    ranges.emplace_back(index, index + 1);

  MultidimensionalRange range(ranges);

  if (groupIt == groups.end())
    groups.emplace_back(MCIS(std::move(range)), std::move(delta));
  else
    groupIt->addEquation(std::move(range));
}

/**
 * Get the index to be used to access a flattened array.
 * If an array is declared as [a][b][c], then the access [i][j][k] corresponds
 * to the access [k + c * (j + b * (i))] of the flattened array of size
 * [a * b * c].
 *
 * @param dimensions 	original array dimensions
 * @param indexes 		access with respect to the original dimensions
 * @return flattened index
 */
static size_t flattenIndexes(llvm::ArrayRef<size_t> dimensions, llvm::ArrayRef<size_t> indexes)
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

class FlatMCIM : public MCIM::Impl
{
  public:
  FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

  void apply(const AccessFunction& access) override;
  bool get(llvm::ArrayRef<long> indexes) const override;
  void set(llvm::ArrayRef<long> indexes) override;

  bool empty() const override;
  void clear() override;

  MCIS flattenEquations() const override;

  MCIS flattenVariables() const override;

  MCIM filterEquations(const MCIS& filter) const override;

  MCIM filterVariables(const MCIS& filter) const override;

  private:
};

FlatMCIM::FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() != getVariableRanges().rank());
}

void FlatMCIM::apply(const AccessFunction& access)
{
  // TODO
}

bool FlatMCIM::get(llvm::ArrayRef<long> indexes) const
{
  // TODO
}

void FlatMCIM::set(llvm::ArrayRef<long> indexes)
{
  // TODO
}

bool FlatMCIM::empty() const
{
  // TODO
}

void FlatMCIM::clear()
{
  // TODO
}

MCIS FlatMCIM::flattenEquations() const
{
  // TODO
}

MCIS FlatMCIM::flattenVariables() const
{
  //TODO
}

MCIM FlatMCIM::filterEquations(const MCIS& filter) const
{
  // TODO
}

MCIM FlatMCIM::filterVariables(const MCIS& filter) const
{
  // TODO
}

MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
{
  if (equationRanges.rank() == variableRanges.rank())
    impl = std::make_unique<RegularMCIM>(std::move(equationRanges), std::move(variableRanges));
  else
    impl = std::make_unique<FlatMCIM>(std::move(equationRanges), std::move(variableRanges));
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
  return impl->filterEquations(filter);
}

MCIM MCIM::filterVariables(const MCIS& filter) const
{
  return impl->filterVariables(filter);
}
