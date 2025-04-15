#include "marco/Modeling/MCIMGroupEmpty.h"

using namespace ::marco::modeling;

namespace marco::modeling {
std::unique_ptr<MCIMGroup>
MCIMGroupEmpty::build(const AccessFunctionEmpty &accessFunction) {
  return std::make_unique<MCIMGroupEmpty>(accessFunction);
}

MCIMGroupEmpty::MCIMGroupEmpty(const AccessFunction &accessFunction)
    : MCIMGroupGeneric(accessFunction) {}

MCIMGroupEmpty::MCIMGroupEmpty(const MCIMGroupEmpty &other)
    : MCIMGroupGeneric(other) {}

MCIMGroupEmpty::~MCIMGroupEmpty() = default;

std::unique_ptr<MCIMGroup> MCIMGroupEmpty::clone() const {
  return std::make_unique<MCIMGroupEmpty>(*this);
}

IndexSet MCIMGroupEmpty::getValues() const { return IndexSet(Point(0)); }

bool MCIMGroupEmpty::hasValue(const Point &point) const {
  return getValues().contains(point);
}

void MCIMGroupEmpty::removeValues(const IndexSet &removedValues) {
  if (removedValues.contains(Point(0))) {
    keys.clear();
  }
}

bool MCIMGroupEmpty::has(const Point &key, const Point &value) const {
  return getKeys().contains(key) && value == Point(0);
}

bool MCIMGroupEmpty::set(const Point &key, const Point &value) {
  if (value == Point(0)) {
    addKeys(IndexSet(key));
    return true;
  }

  return false;
}

bool MCIMGroupEmpty::unset(const Point &key, const Point &value) {
  if (value == Point(0)) {
    removeKeys(IndexSet(key));
    return true;
  }

  return false;
}

std::unique_ptr<MCIMGroup>
MCIMGroupEmpty::filterValues(const IndexSet &filter) const {
  auto result = MCIMGroupGeneric::build(getAccessFunction());

  if (filter.contains(Point(0))) {
    result->addKeys(getKeys());
  }

  return result;
}
} // namespace marco::modeling
