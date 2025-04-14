#include "marco/Modeling/MCIMGroupAffineConstant.h"

using namespace ::marco::modeling;

namespace marco::modeling {
std::unique_ptr<MCIMGroup> MCIMGroupAffineConstant::build(
    const AccessFunctionAffineConstant &accessFunction) {
  return std::make_unique<MCIMGroupAffineConstant>(accessFunction);
}

MCIMGroupAffineConstant::MCIMGroupAffineConstant(
    const AccessFunctionAffineConstant &accessFunction)
    : MCIMGroupGeneric(accessFunction) {
  if (auto mappedPoint = accessFunction.getMappedPoint()) {
    mapped += *mappedPoint;
  }
}

MCIMGroupAffineConstant::MCIMGroupAffineConstant(
    const MCIMGroupAffineConstant &other)
    : MCIMGroupGeneric(other), mapped(other.mapped) {}

MCIMGroupAffineConstant::~MCIMGroupAffineConstant() = default;

std::unique_ptr<MCIMGroup> MCIMGroupAffineConstant::clone() const {
  return std::make_unique<MCIMGroupAffineConstant>(*this);
}

bool MCIMGroupAffineConstant::hasValue(const Point &point) const {
  return mapped.contains(point);
}

void MCIMGroupAffineConstant::removeValues(const IndexSet &removedValues) {
  if (mapped.overlaps(removedValues)) {
    keys.clear();
  }
}

bool MCIMGroupAffineConstant::has(const Point &key, const Point &value) const {
  return keys.contains(key) && mapped.contains(value);
}

bool MCIMGroupAffineConstant::set(const Point &key, const Point &value) {
  if (mapped.contains(value)) {
    addKeys(IndexSet(key));
    return true;
  }

  return false;
}

bool MCIMGroupAffineConstant::unset(const Point &key, const Point &value) {
  if (mapped.contains(value)) {
    removeKeys(IndexSet(key));
    return true;
  }

  return false;
}

std::unique_ptr<MCIMGroup>
MCIMGroupAffineConstant::filterValues(const IndexSet &filter) const {
  auto result = MCIMGroupGeneric::build(getAccessFunction());

  if (filter.contains(mapped)) {
    result->addKeys(keys);
  }

  return result;
}
} // namespace marco::modeling
