#include <marco/matching/MCIM.h>
#include <numeric>

using namespace marco::matching;

class Delta
{
  public:
  Delta(llvm::ArrayRef<long> values);

  Delta(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes);

  bool operator==(const Delta& other) const;
  bool operator!=(const Delta& other) const;

  private:
  llvm::SmallVector<long, 3> values;
};

Delta::Delta(llvm::ArrayRef<long> values)
        : values(values.begin(), values.end())
{
}

Delta::Delta(llvm::ArrayRef<long> equationIndexes, llvm::ArrayRef<long> variableIndexes)
{
  assert(equationIndexes.size() == variableIndexes.size());

  for (const auto& [equationIndex, variableIndex] : llvm::zip(equationIndexes, variableIndexes))
    values.push_back(variableIndex - equationIndex);
}

bool Delta::operator==(const Delta &other) const
{
  return llvm::all_of(llvm::zip(values, other.values), [](const auto& pair) {
    return std::get<0>(pair) == std::get<1>(pair);
  });
}

bool Delta::operator!=(const Delta &other) const
{
  return llvm::any_of(llvm::zip(values, other.values), [](const auto& pair) {
      return std::get<0>(pair) != std::get<1>(pair);
  });
}

class MCIMElement
{
  public:
  MCIMElement(MCIS k, Delta delta);

  bool operator==(const MCIMElement& other) const;
  bool operator!=(const MCIMElement& other) const;

  MCIS& getEquationIndexes();
  const MCIS& getEquationIndexes() const;

  const Delta& getDelta() const;

  private:
  MCIS k;
  Delta delta;
};

MCIMElement::MCIMElement(MCIS k, Delta delta)
        : k(std::move(k)), delta(std::move(delta))
{
}

bool MCIMElement::operator==(const MCIMElement &other) const
{
  if (k != other.k)
    return false;

  if (delta != other.delta)
    return false;

  return true;
}

bool MCIMElement::operator!=(const MCIMElement &other) const
{
  if (k == other.k)
    return false;

  if (delta == other.delta)
    return false;

  return true;
}

MCIS& MCIMElement::getEquationIndexes()
{
  return k;
}

const MCIS& MCIMElement::getEquationIndexes() const
{
  return k;
}

const Delta &MCIMElement::getDelta() const
{
  return delta;
}

namespace marco::matching
{
  class MCIM::Impl
  {
    public:
    Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    const MultidimensionalRange& getEquationRanges() const;
    const MultidimensionalRange& getVariableRanges() const;

    virtual bool empty() const = 0;
    virtual void clear() = 0;

    virtual void apply(AccessFunction access) = 0;

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
  RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

  bool empty() const override;
  void clear() override;

  void apply(AccessFunction access) override;

  private:
  llvm::SmallVector<MCIMElement, 3> groups;
};

RegularMCIM::RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() == getVariableRanges().rank());
}

bool RegularMCIM::empty() const
{
  return groups.empty();
}

void RegularMCIM::clear()
{
  groups.clear();
}

void RegularMCIM::apply(AccessFunction access)
{
  for (const auto& equationIndexes : getEquationRanges())
  {
    assert(access.size() == getVariableRanges().rank());

    llvm::SmallVector<long, 6> variableIndexes;
    access.map(variableIndexes, variableIndexes);
    Delta delta(equationIndexes, variableIndexes);

    auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
    });

    if (groupIt == groups.end())
    {

    }
    else
    {

    }
  }
}

/**
 * Get the index to be used to access a flattened array.
 * If an array is declared as [a][b][c], then the access [i][j][k] corresponds
 * to the access [k + c * (j + b * (i))] of the flattened array of size
 * [a * b * c].
 *
 * @param dimensions 	original array dimensions
 * @param indexes 		access with respect to the original dimensions
 * @return flattened array index
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

  bool empty() const override;
  void clear() override;

  void apply(AccessFunction access) override;

  private:

};

FlatMCIM::FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() != getVariableRanges().rank());
}

bool FlatMCIM::empty() const
{
  // TODO
}

void FlatMCIM::clear()
{
  // TODO
}

void FlatMCIM::apply(AccessFunction access)
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

void MCIM::apply(AccessFunction access)
{
  impl->apply(std::move(access));
}

bool MCIM::empty() const
{
  return impl->empty();
}

void MCIM::clear()
{
  impl->clear();
}
