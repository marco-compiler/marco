#include <llvm/Support/Casting.h>
#include <marco/matching/MCIM.h>
#include <numeric>
#include <type_traits>

using namespace marco::matching;
using namespace marco::matching::detail;

MCIM::IndexesIterator::IndexesIterator(
        const MultidimensionalRange& equationRange,
        const MultidimensionalRange& variableRange,
        std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction)
        : eqCurrentIt(initFunction(equationRange)),
          eqEndIt(equationRange.end()),
          varBeginIt(variableRange.begin()),
          varCurrentIt(initFunction(variableRange)),
          varEndIt(variableRange.end())
{
  if (eqCurrentIt != eqEndIt)
  {
    assert(varCurrentIt != varEndIt);
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

MCIM::IndexesIterator::value_type MCIM::IndexesIterator::operator*() const
{
  return std::make_pair(*eqCurrentIt, *varCurrentIt);
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

    varCurrentIt = varBeginIt;
  }
}

namespace marco::matching::detail
{
  class MCIM::Impl
  {
    public:
    enum MCIMKind
    {
      Regular,
      Flat
    };

    Impl(MCIMKind kind, MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    MCIMKind getKind() const
    {
      return kind;
    }

    template<typename T>
    bool isa() const
    {
      return llvm::isa<T>(this);
    }

    template<typename T>
    T* dyn_cast()
    {
      return llvm::dyn_cast<T>(this);
    }

    template<typename T>
    const T* dyn_cast() const
    {
      return llvm::dyn_cast<T>(this);
    }

    virtual bool operator==(const MCIM::Impl& rhs) const;
    virtual bool operator!=(const MCIM::Impl& rhs) const;

    virtual std::unique_ptr<MCIM::Impl> clone() = 0;

    const MultidimensionalRange& getEquationRanges() const;
    const MultidimensionalRange& getVariableRanges() const;

    llvm::iterator_range<IndexesIterator> getIndexes() const;

    virtual MCIM::Impl& operator+=(const MCIM::Impl& rhs);
    virtual MCIM::Impl& operator-=(const MCIM::Impl& rhs);

    virtual void apply(const AccessFunction& access) = 0;

    virtual bool get(const Point& equation, const Point& variable) const = 0;
    virtual void set(const Point& equation, const Point& variable) = 0;
    virtual void unset(const Point& equation, const Point& variable) = 0;

    virtual bool empty() const = 0;
    virtual void clear() = 0;

    virtual MCIS flattenEquations() const = 0;
    virtual MCIS flattenVariables() const = 0;

    virtual std::unique_ptr<MCIM::Impl> filterEquations(const MCIS& filter) const = 0;
    virtual std::unique_ptr<MCIM::Impl> filterVariables(const MCIS& filter) const = 0;

    virtual std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const = 0;

    private:
    const MCIMKind kind;
    MultidimensionalRange equationRanges;
    MultidimensionalRange variableRanges;
  };
}

MCIM::Impl::Impl(MCIMKind kind, MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
    : kind(kind), equationRanges(std::move(equationRanges)), variableRanges(std::move(variableRanges))
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

bool MCIM::Impl::operator==(const MCIM::Impl& rhs) const
{
  for (const auto& [equation, variable] : getIndexes())
    if (get(equation, variable) != rhs.get(equation, variable))
      return false;

  return true;
}

bool MCIM::Impl::operator!=(const MCIM::Impl& rhs) const
{
  for (const auto& [equation, variable] : getIndexes())
    if (get(equation, variable) != rhs.get(equation, variable))
      return true;

  return false;
}

llvm::iterator_range<MCIM::IndexesIterator> MCIM::Impl::getIndexes() const
{
  IndexesIterator begin(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.begin();
  });

  IndexesIterator end(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.end();
  });

  return llvm::iterator_range<MCIM::IndexesIterator>(begin, end);
}

MCIM::Impl& MCIM::Impl::operator+=(const MCIM::Impl& rhs)
{
  for (const auto& [equation, variable] : getIndexes())
    if (rhs.get(equation, variable))
      set(equation, variable);

  return *this;
}

MCIM::Impl& MCIM::Impl::operator-=(const MCIM::Impl& rhs)
{
  for (const auto& [equation, variable] : getIndexes())
    set(equation, variable);

  for (const auto& [equation, variable] : getIndexes())
    if (rhs.get(equation, variable))
      unset(equation, variable);

  return *this;
}

class RegularMCIM : public MCIM::Impl
{
  public:
  class Delta
  {
    public:
    Delta(const Point& keys, const Point& values);

    bool operator==(const Delta& other) const;
    long operator[](size_t index) const;

    size_t size() const;

    Delta inverse() const;

    private:
    llvm::SmallVector<Point::data_type, 3> values;
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

  static bool classof(const MCIM::Impl* obj)
  {
    return obj->getKind() == Regular;
  }

  bool operator==(const MCIM::Impl& rhs) const override;
  bool operator!=(const MCIM::Impl& rhs) const override;

  std::unique_ptr<MCIM::Impl> clone() override;

  MCIM::Impl& operator+=(const MCIM::Impl& rhs) override;
  MCIM::Impl& operator-=(const MCIM::Impl& rhs) override;

  void apply(const AccessFunction& access) override;

  bool get(const Point& equation, const Point& variable) const override;
  void set(const Point& equation, const Point& variable) override;
  void unset(const Point& equation, const Point& variable) override;

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

RegularMCIM::Delta::Delta(const Point& keys, const Point& values)
{
  assert(keys.rank() == values.rank());

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
  keys += std::move(newKeys);
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

      result += MultidimensionalRange(valueRanges);
    }
  }

  return result;
}

RegularMCIM::MCIMElement RegularMCIM::MCIMElement::inverse() const
{
  return RegularMCIM::MCIMElement(getValues(), delta.inverse());
}

RegularMCIM::RegularMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(Regular, std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() == getVariableRanges().rank());
}

bool RegularMCIM::operator==(const MCIM::Impl& rhs) const
{
  if (auto other = rhs.dyn_cast<RegularMCIM>())
  {
    if (groups.empty() && other->groups.empty())
      return true;

    if (groups.size() != other->groups.size())
      return false;

    for (const auto& group : other->groups)
    {
      auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
        return obj.getDelta() == group.getDelta();
      });

      if (groupIt == groups.end())
        return false;

      if (group.getKeys() != groupIt->getKeys())
        return false;
    }

    return true;
  }

  return MCIM::Impl::operator==(rhs);
}

bool RegularMCIM::operator!=(const MCIM::Impl& rhs) const
{
  if (auto other = rhs.dyn_cast<RegularMCIM>())
  {
    if (groups.empty() && other->groups.empty())
      return false;

    if (groups.size() != other->groups.size())
      return true;

    for (const auto& group : other->groups)
    {
      auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
        return obj.getDelta() == group.getDelta();
      });

      if (groupIt == groups.end())
        return true;

      if (group.getKeys() != groupIt->getKeys())
        return true;
    }

    return false;
  }

  return MCIM::Impl::operator==(rhs);
}

std::unique_ptr<MCIM::Impl> RegularMCIM::clone()
{
  return std::make_unique<RegularMCIM>(*this);
}

MCIM::Impl& RegularMCIM::operator+=(const MCIM::Impl& rhs)
{
  if (auto other = rhs.dyn_cast<RegularMCIM>())
  {
    for (const auto& group : other->groups)
      add(group.getKeys(), group.getDelta());

    return *this;
  }

  return MCIM::Impl::operator+=(rhs);
}

MCIM::Impl& RegularMCIM::operator-=(const MCIM::Impl& rhs)
{
  if (auto other = rhs.dyn_cast<RegularMCIM>())
  {
    llvm::SmallVector<MCIMElement, 3> newGroups;

    for (const auto& group : groups)
    {
      auto groupIt = llvm::find_if(other->groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
      });

      if (groupIt == other->groups.end())
      {
        newGroups.push_back(std::move(group));
      }
      else
      {
        MCIS diff = group.getKeys() - groupIt->getKeys();
        newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
      }
    }

    groups = std::move(newGroups);
    return *this;
  }

  return MCIM::Impl::operator+=(rhs);
}

void RegularMCIM::apply(const AccessFunction& access)
{
  for (const auto& equationIndexes : getEquationRanges())
  {
    assert(access.size() == getVariableRanges().rank());
    auto variableIndexes = access.map(equationIndexes);
    set(equationIndexes, variableIndexes);
  }
}

bool RegularMCIM::get(const Point& equation, const Point& variable) const
{
  assert(equation.rank() == getEquationRanges().rank());
  assert(variable.rank() == getVariableRanges().rank());

  Delta delta(equation, variable);

  return llvm::any_of(groups, [&](const MCIMElement& group) -> bool {
      return group.getDelta() == delta && group.getKeys().contains(equation);
  });
}

void RegularMCIM::set(const Point& equation, const Point& variable)
{
  assert(equation.rank() == getEquationRanges().rank());
  assert(variable.rank() == getVariableRanges().rank());

  Delta delta(equation, variable);
  llvm::SmallVector<Range, 3> ranges;

  for (const auto& index : equation)
    ranges.emplace_back(index, index + 1);

  MCIS keys(MultidimensionalRange(std::move(ranges)));
  add(std::move(keys), std::move(delta));
}

void RegularMCIM::unset(const Point& equation, const Point& variable)
{
  assert(equation.rank() == getEquationRanges().rank());
  assert(variable.rank() == getVariableRanges().rank());

  llvm::SmallVector<Range, 3> ranges;

  for (size_t i = 0; i < equation.rank(); ++i)
    ranges.emplace_back(equation[i], equation[i] + 1);

  llvm::SmallVector<MCIMElement, 3> newGroups;

  for (const auto& group : groups)
  {
    MCIS diff = group.getKeys() - MultidimensionalRange(std::move(ranges));

    if (!diff.empty())
      newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
  }

  groups = std::move(newGroups);
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
    result += group.getValues();

  return result;
}

MCIS RegularMCIM::flattenVariables() const
{
  MCIS result;

  for (const auto& group : groups)
    result += group.getKeys();

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
    using data_type = std::make_unsigned_t<Point::data_type>;

    Delta(data_type key, data_type value);

    bool operator==(const Delta& other) const;

    std::make_signed_t<data_type> getValue() const;
    Delta inverse() const;

    private:
    std::make_signed_t<data_type> value;
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

  static bool classof(const MCIM::Impl* obj)
  {
    return obj->getKind() == Flat;
  }

  bool operator==(const MCIM::Impl& rhs) const override;
  bool operator!=(const MCIM::Impl& rhs) const override;

  std::unique_ptr<MCIM::Impl> clone() override;

  MCIM::Impl& operator+=(const MCIM::Impl& rhs) override;
  MCIM::Impl& operator-=(const MCIM::Impl& rhs) override;

  void apply(const AccessFunction& access) override;

  bool get(const Point& equation, const Point& variable) const override;
  void set(const Point& equation, const Point& variable) override;
  void unset(const Point& equation, const Point& variable) override;

  bool empty() const override;
  void clear() override;

  MCIS flattenEquations() const override;
  MCIS flattenVariables() const override;

  std::unique_ptr<MCIM::Impl> filterEquations(const MCIS& filter) const override;
  std::unique_ptr<MCIM::Impl> filterVariables(const MCIS& filter) const override;

  std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const override;

  private:
  Point getFlatEquation(const Point& equation) const;
  Point getFlatVariable(const Point& variable) const;

  void add(MCIS keys, Delta delta);

  llvm::SmallVector<MCIMElement, 3> groups;

  // Stored for faster lookup
  llvm::SmallVector<size_t, 3> equationDimensions;
  llvm::SmallVector<size_t, 3> variableDimensions;
};

template<typename T = Point::data_type>
static void convertIndexesToZeroBased(
        llvm::ArrayRef<T> indexes,
        const MultidimensionalRange& base,
        llvm::SmallVectorImpl<std::make_unsigned_t<T>>& rescaled)
{
  assert(base.rank() == indexes.size());

  for (unsigned int i = 0, e = base.rank(); i < e; ++i)
  {
    const auto& monoDimRange = base[i];
    T index = indexes[i] - monoDimRange.getBegin();
    assert(index >= 0 && index < monoDimRange.getEnd() - monoDimRange.getBegin());
    rescaled.push_back(std::move(index));
  }
}

template<typename T = Point::data_type>
static void convertIndexesFromZeroBased(
        llvm::ArrayRef<std::make_unsigned_t<T>> indexes,
        const MultidimensionalRange& base,
        llvm::SmallVectorImpl<T>& rescaled)
{
  assert(base.rank() == indexes.size());

  for (size_t i = 0, e = base.rank(); i < e; ++i)
  {
    const auto& monoDimRange = base[i];
    T index = indexes[i] + monoDimRange.getBegin();
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
template<typename T = Point::data_type>
static std::make_unsigned_t<T> flattenIndexes(
        llvm::ArrayRef<std::make_unsigned_t<T>> indexes,
        llvm::ArrayRef<size_t> dimensions)
{
  assert(dimensions.size() == indexes.size());
  std::make_unsigned_t<T> result = 0;

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
template<typename T = Point::data_type>
static void unflattenIndex(
        llvm::ArrayRef<size_t> dimensions,
        std::make_unsigned_t<T> index,
        llvm::SmallVectorImpl<std::make_unsigned_t<T>>& results)
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
    llvm::SmallVector<Point::data_type, 3> firstItemIndexes;
    llvm::SmallVector<Point::data_type, 3> lastItemIndexes;

    for (size_t i = 0, e = multiDimRange.rank(); i < e; ++i)
    {
      const auto& monoDimRange = multiDimRange[i];
      firstItemIndexes.push_back(monoDimRange.getBegin());
      lastItemIndexes.push_back(monoDimRange.getEnd() - 1);
    }

    llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> firstItemRescaled;
    convertIndexesToZeroBased<Point::data_type>(firstItemIndexes, range, firstItemRescaled);

    llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> lastItemRescaled;
    convertIndexesToZeroBased<Point::data_type>(lastItemIndexes, range, lastItemRescaled);

    auto firstItemFlattened = flattenIndexes(firstItemRescaled, dimensions);
    auto lastItemFlattened = flattenIndexes(lastItemRescaled, dimensions);

    result += MultidimensionalRange(Range(firstItemFlattened, lastItemFlattened + 1));
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

    std::make_unsigned_t<Point::data_type> firstItemFlattened = monoDimRange.getBegin();
    std::make_unsigned_t<Point::data_type> lastItemFlattened = monoDimRange.getEnd() - 1;

    for (auto flattened = firstItemFlattened; flattened <= lastItemFlattened; ++flattened)
    {
      llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
      unflattenIndex(dimensions, flattened, rescaled);

      llvm::SmallVector<Point::data_type, 3> indexes;
      convertIndexesFromZeroBased(rescaled, range, indexes);

      llvm::SmallVector<Range, 3> ranges;

      for (const auto& index : indexes)
        ranges.emplace_back(index, index + 1);

      result += MultidimensionalRange(std::move(ranges));
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
  keys += std::move(newKeys);
}

const FlatMCIM::Delta& FlatMCIM::MCIMElement::getDelta() const
{
  return delta;
}

MCIS FlatMCIM::MCIMElement::getValues() const
{
  MCIS result;

  for (const auto& keyRange : keys)
  {
    llvm::SmallVector<Range, 3> valueRanges;

    for (size_t i = 0, e = keyRange.rank(); i < e; ++i)
      valueRanges.emplace_back(keyRange[i].getBegin() + delta.getValue(), keyRange[i].getEnd() + delta.getValue());

    result += MultidimensionalRange(valueRanges);
  }

  return result;
}

FlatMCIM::MCIMElement FlatMCIM::MCIMElement::inverse() const
{
  return FlatMCIM::MCIMElement(getValues(), delta.inverse());
}

FlatMCIM::FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
        : MCIM::Impl(Flat, std::move(equationRanges), std::move(variableRanges))
{
  assert(getEquationRanges().rank() != getVariableRanges().rank());

  for (size_t i = 0, e = getEquationRanges().rank(); i < e; ++i)
    equationDimensions.push_back(getEquationRanges()[i].size());

  for (size_t i = 0, e = getVariableRanges().rank(); i < e; ++i)
    variableDimensions.push_back(getVariableRanges()[i].size());
}

bool FlatMCIM::operator==(const MCIM::Impl& rhs) const
{
  if (auto other = rhs.dyn_cast<FlatMCIM>())
  {
    if (groups.empty() && other->groups.empty())
      return true;

    if (groups.size() != other->groups.size())
      return false;

    for (const auto& group : other->groups)
    {
      auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
      });

      if (groupIt == groups.end())
        return false;

      if (group.getKeys() != groupIt->getKeys())
        return false;
    }

    return true;
  }

  return MCIM::Impl::operator==(rhs);
}

bool FlatMCIM::operator!=(const MCIM::Impl& rhs) const
{
  if (auto other = rhs.dyn_cast<FlatMCIM>())
  {
    if (groups.empty() && other->groups.empty())
      return false;

    if (groups.size() != other->groups.size())
      return true;

    for (const auto& group : other->groups)
    {
      auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
      });

      if (groupIt == groups.end())
        return true;

      if (group.getKeys() != groupIt->getKeys())
        return true;
    }

    return false;
  }

  return MCIM::Impl::operator==(rhs);
}

std::unique_ptr<MCIM::Impl> FlatMCIM::clone()
{
  return std::make_unique<FlatMCIM>(*this);
}

MCIM::Impl& FlatMCIM::operator+=(const MCIM::Impl& rhs)
{
  if (auto other = rhs.dyn_cast<FlatMCIM>())
  {
    for (const auto& group : other->groups)
      add(group.getKeys(), group.getDelta());

    return *this;
  }

  return MCIM::Impl::operator+=(rhs);
}

MCIM::Impl& FlatMCIM::operator-=(const MCIM::Impl& rhs)
{
  if (auto other = rhs.dyn_cast<FlatMCIM>())
  {
    llvm::SmallVector<MCIMElement, 3> newGroups;

    for (const auto& group : groups)
    {
      auto groupIt = llvm::find_if(other->groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
      });

      if (groupIt == other->groups.end())
      {
        newGroups.push_back(std::move(group));
      }
      else
      {
        MCIS diff = group.getKeys() - groupIt->getKeys();
        newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
      }
    }

    groups = std::move(newGroups);
    return *this;
  }

  return MCIM::Impl::operator+=(rhs);
}

void FlatMCIM::apply(const AccessFunction& access)
{
  for (const auto& equation : getEquationRanges())
  {
    assert(access.size() == getVariableRanges().rank());
    auto variable = access.map(equation);
    set(equation, variable);
  }
}

bool FlatMCIM::get(const Point& equation, const Point& variable) const
{
  auto flatEquation = getFlatEquation(equation);
  auto flatVariable = getFlatVariable(variable);

  Delta delta(flatEquation[0], flatVariable[0]);

  return llvm::any_of(groups, [&](const MCIMElement& group) -> bool {
      return group.getDelta() == delta && group.getKeys().contains(flatEquation);
  });
}

void FlatMCIM::set(const Point& equation, const Point& variable)
{
  auto flatEquation = getFlatEquation(equation);
  auto flatVariable = getFlatVariable(variable);

  Delta delta(flatEquation[0], flatVariable[0]);
  MCIS keys(MultidimensionalRange(Range(flatEquation[0], flatEquation[0] + 1)));
  add(std::move(keys), std::move(delta));
}

void FlatMCIM::unset(const Point& equation, const Point& variable)
{
  auto flatEquation = getFlatEquation(equation);

  llvm::SmallVector<MCIMElement, 3> newGroups;

  for (const auto& group : groups)
  {
    MCIS diff = group.getKeys() - MultidimensionalRange(Range(flatEquation[0], flatEquation[0] + 1));

    if (!diff.empty())
      newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
  }

  groups = std::move(newGroups);
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
    result += group.getValues();

  return unflattenMCIS(result, getVariableRanges(), variableDimensions);
}

MCIS FlatMCIM::flattenVariables() const
{
  MCIS result;

  for (const auto& group : groups)
    result += group.getKeys();

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

Point FlatMCIM::getFlatEquation(const Point& equation) const
{
  assert(equation.rank() == getEquationRanges().rank());

  llvm::SmallVector<Point::data_type, 3> indexes(equation.begin(), equation.end());

  llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
  convertIndexesToZeroBased<Point::data_type>(indexes, getEquationRanges(), rescaled);

  auto flattened = flattenIndexes(rescaled, equationDimensions);
  return Point(flattened);
}

Point FlatMCIM::getFlatVariable(const Point& variable) const
{
  assert(variable.rank() == getVariableRanges().rank());

  llvm::SmallVector<Point::data_type, 3> indexes(variable.begin(), variable.end());

  llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
  convertIndexesToZeroBased<Point::data_type>(indexes, getVariableRanges(), rescaled);

  auto flattened = flattenIndexes(rescaled, variableDimensions);
  return Point(flattened);
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

MCIM::MCIM(MCIM&& other) = default;

MCIM::~MCIM() = default;

MCIM& MCIM::operator=(const MCIM& other)
{
  MCIM result(other);
  swap(*this, result);
  return *this;
}

namespace marco::matching::detail
{
  void swap(MCIM& first, MCIM& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }
}

bool MCIM::operator==(const MCIM& other) const
{
  if (getEquationRanges() != other.getEquationRanges())
    return false;

  if (getVariableRanges() != other.getVariableRanges())
    return false;

  return (*impl) == *other.impl;
}

bool MCIM::operator!=(const MCIM& other) const
{
  if (getEquationRanges() != other.getEquationRanges())
    return true;

  if (getVariableRanges() != other.getVariableRanges())
    return true;

  return (*impl) != *other.impl;
}

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
  return impl->getIndexes();
}

MCIM& MCIM::operator+=(const MCIM& rhs)
{
  assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
  assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

  (*impl) += *rhs.impl;
  return *this;
}

MCIM MCIM::operator+(const MCIM& rhs) const
{
  MCIM result = *this;
  result += rhs;
  return result;
}

MCIM& MCIM::operator-=(const MCIM& rhs)
{
  assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
  assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

  (*impl) -= *rhs.impl;
  return *this;
}

MCIM MCIM::operator-(const MCIM& rhs) const
{
  MCIM result = *this;
  result -= rhs;
  return result;
}

void MCIM::apply(const AccessFunction& access)
{
  impl->apply(access);
}

bool MCIM::get(const Point& equation, const Point& variable) const
{
  return impl->get(equation, variable);
}

void MCIM::set(const Point& equation, const Point& variable)
{
  impl->set(equation, variable);
}

void MCIM::unset(const Point& equation, const Point& variable)
{
  impl->unset(equation, variable);
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

static size_t getIndexesWidth(const Point& indexes)
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

      stream << variableIndexes;
    }

    // The first line containing the variable indexes is finished
    stream << "\n";

    // Print a line for each equation
    for (const auto& equation : equationRanges)
    {
      for (size_t i = getIndexesWidth(equation); i < equationIndexesMaxWidth; ++i)
        stream << " ";

      stream << equation;

      for (const auto& variable : variableRanges)
      {
        stream << " ";

        size_t columnWidth = variableIndexesColumnWidth;
        size_t spacesAfter = (columnWidth - 1) / 2;
        size_t spacesBefore = columnWidth - 1 - spacesAfter;

        for (size_t i = 0; i < spacesBefore; ++i)
          stream << " ";

        stream << (mcim.get(equation, variable) ? 1 : 0);

        for (size_t i = 0; i < spacesAfter; ++i)
          stream << " ";
      }

      stream << "\n";
    }

    return stream;
  }
}
