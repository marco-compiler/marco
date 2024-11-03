#include "marco/Modeling/MCIMGroupGeneric.h"

using namespace ::marco::modeling;

namespace marco::modeling {
std::unique_ptr<MCIMGroup>
MCIMGroupGeneric::build(const AccessFunction &accessFunction) {
  return std::make_unique<MCIMGroupGeneric>(accessFunction);
}

MCIMGroupGeneric::MCIMGroupGeneric(const AccessFunction &accessFunction)
    : MCIMGroup(accessFunction) {}

MCIMGroupGeneric::MCIMGroupGeneric(const MCIMGroupGeneric &other)
    : MCIMGroup(other), keys(other.keys) {}

MCIMGroupGeneric::~MCIMGroupGeneric() = default;

std::unique_ptr<MCIMGroup> MCIMGroupGeneric::clone() const {
  return std::make_unique<MCIMGroupGeneric>(*this);
}

bool MCIMGroupGeneric::empty() const { return keys.empty(); }

IndexSet MCIMGroupGeneric::getKeys() const { return keys; }

bool MCIMGroupGeneric::hasKey(const Point &key) const {
  return keys.contains(key);
}

void MCIMGroupGeneric::addKeys(const IndexSet &newKeys) { keys += newKeys; }

void MCIMGroupGeneric::removeKeys(const IndexSet &removedKeys) {
  keys -= removedKeys;
}

IndexSet MCIMGroupGeneric::getValues() const {
  return getAccessFunction().map(keys);
}

bool MCIMGroupGeneric::hasValue(const Point &point) const {
  return getValues().contains(point);
}

bool MCIMGroupGeneric::has(const Point &key, const Point &value) const {
  return keys.contains(key) && getAccessFunction().map(key).contains(value);
}

bool MCIMGroupGeneric::set(const Point &key, const Point &value) {
  IndexSet inverseKeys =
      getAccessFunction().inverseMap(IndexSet(value), IndexSet(key));
  addKeys(inverseKeys);
  return !inverseKeys.empty();
}

bool MCIMGroupGeneric::unset(const Point &key, const Point &value) {
  IndexSet inverseKeys =
      getAccessFunction().inverseMap(IndexSet(value), IndexSet(key));
  removeKeys(inverseKeys);
  return !inverseKeys.empty();
}

std::unique_ptr<MCIMGroup>
MCIMGroupGeneric::filterKeys(const IndexSet &filter) const {
  auto result = MCIMGroupGeneric::build(getAccessFunction());
  result->addKeys(keys.intersect(filter));
  return result;
}

std::unique_ptr<MCIMGroup>
MCIMGroupGeneric::filterValues(const IndexSet &filter) const {
  auto result = MCIMGroupGeneric::build(getAccessFunction());
  result->addKeys(
      getAccessFunction().inverseMap(getValues().intersect(filter), keys));
  return result;
}
} // namespace marco::modeling
