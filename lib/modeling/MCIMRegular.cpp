#include "marco/modeling/MCIMRegular.h"

namespace marco::modeling::internal
{
  RegularMCIM::Delta::Delta(const Point& keys, const Point& values)
  {
    assert(keys.rank() == values.rank());

    for (const auto&[key, value]: llvm::zip(keys, values)) {
      this->values.push_back(value - key);
    }
  }

  bool RegularMCIM::Delta::operator==(const Delta& other) const
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

    for (auto& value: result.values) {
      value *= -1;
    }

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

    for (const auto& keyRange: keys) {
      llvm::SmallVector<Range, 3> valueRanges;

      for (size_t i = 0, e = keyRange.rank(); i < e; ++i) {
        valueRanges.emplace_back(keyRange[i].getBegin() + delta[i], keyRange[i].getEnd() + delta[i]);
      }

      result += MultidimensionalRange(valueRanges);
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
    if (auto other = rhs.dyn_cast<RegularMCIM>()) {
      if (groups.empty() && other->groups.empty()) {
        return true;
      }

      if (groups.size() != other->groups.size()) {
        return false;
      }

      for (const auto& group: other->groups) {
        auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == groups.end()) {
          return false;
        }

        if (group.getKeys() != groupIt->getKeys()) {
          return false;
        }
      }

      return true;
    }

    return MCIM::Impl::operator==(rhs);
  }

  bool RegularMCIM::operator!=(const MCIM::Impl& rhs) const
  {
    if (auto other = rhs.dyn_cast<RegularMCIM>()) {
      if (groups.empty() && other->groups.empty()) {
        return false;
      }

      if (groups.size() != other->groups.size()) {
        return true;
      }

      for (const auto& group: other->groups) {
        auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == groups.end()) {
          return true;
        }

        if (group.getKeys() != groupIt->getKeys()) {
          return true;
        }
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
    if (auto other = rhs.dyn_cast<RegularMCIM>()) {
      for (const auto& group: other->groups) {
        add(group.getKeys(), group.getDelta());
      }

      return *this;
    }

    return MCIM::Impl::operator+=(rhs);
  }

  MCIM::Impl& RegularMCIM::operator-=(const MCIM::Impl& rhs)
  {
    if (auto other = rhs.dyn_cast<RegularMCIM>()) {
      llvm::SmallVector<MCIMElement, 3> newGroups;

      for (const auto& group: groups) {
        auto groupIt = llvm::find_if(other->groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == other->groups.end()) {
          newGroups.push_back(std::move(group));
        } else {
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
    for (const auto& equationIndexes: getEquationRanges()) {
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

    for (const auto& index: equation) {
      ranges.emplace_back(index, index + 1);
    }

    MCIS keys(MultidimensionalRange(std::move(ranges)));
    add(std::move(keys), std::move(delta));
  }

  void RegularMCIM::unset(const Point& equation, const Point& variable)
  {
    assert(equation.rank() == getEquationRanges().rank());
    assert(variable.rank() == getVariableRanges().rank());

    llvm::SmallVector<Range, 3> ranges;

    for (size_t i = 0; i < equation.rank(); ++i) {
      ranges.emplace_back(equation[i], equation[i] + 1);
    }

    llvm::SmallVector<MCIMElement, 3> newGroups;

    for (const auto& group: groups) {
      MCIS diff = group.getKeys() - MultidimensionalRange(std::move(ranges));

      if (!diff.empty()) {
        newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
      }
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

  MCIS RegularMCIM::flattenRows() const
  {
    MCIS result;

    for (const auto& group: groups) {
      result += group.getValues();
    }

    return result;
  }

  MCIS RegularMCIM::flattenColumns() const
  {
    MCIS result;

    for (const auto& group: groups) {
      result += group.getKeys();
    }

    return result;
  }

  std::unique_ptr<MCIM::Impl> RegularMCIM::filterRows(const MCIS& filter) const
  {
    auto result = std::make_unique<RegularMCIM>(getEquationRanges(), getVariableRanges());

    for (const MCIMElement& group: groups) {
      if (auto& equations = group.getKeys(); equations.overlaps(filter)) {
        result->add(equations.intersect(filter), group.getDelta());
      }
    }

    return result;
  }

  std::unique_ptr<MCIM::Impl> RegularMCIM::filterColumns(const MCIS& filter) const
  {
    auto result = std::make_unique<RegularMCIM>(getEquationRanges(), getVariableRanges());

    for (const auto& group: groups) {
      auto invertedGroup = group.inverse();

      if (auto& variables = invertedGroup.getKeys(); variables.overlaps(filter)) {
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

    for (const auto& group: groups) {
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

    for (const auto& index: equationIndexes) {
      ranges.emplace_back(index, index + 1);
    }

    MCIS keys(MultidimensionalRange(std::move(ranges)));
    add(std::move(keys), std::move(delta));
  }

  void RegularMCIM::add(MCIS keys, Delta delta)
  {
    auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
    });

    if (groupIt == groups.end()) {
      groups.emplace_back(std::move(keys), std::move(delta));
    } else {
      groupIt->addKeys(std::move(keys));
    }
  }
}
