#include "marco/Modeling/MCIMGroup.h"
#include "marco/Modeling/MCIMGroupAffineConstant.h"
#include "marco/Modeling/MCIMGroupEmpty.h"
#include "marco/Modeling/MCIMGroupGeneric.h"

using namespace ::marco::modeling;

namespace marco::modeling {
std::unique_ptr<MCIMGroup>
MCIMGroup::build(const AccessFunction &accessFunction) {
  if (auto casted = accessFunction.dyn_cast<AccessFunctionAffineConstant>()) {
    return MCIMGroupAffineConstant::build(*casted);
  }

  if (auto casted = accessFunction.dyn_cast<AccessFunctionEmpty>()) {
    return MCIMGroupEmpty::build(*casted);
  }

  return MCIMGroupGeneric::build(accessFunction);
}

MCIMGroup::MCIMGroup(const AccessFunction &accessFunction)
    : accessFunction(accessFunction.clone()) {}

MCIMGroup::MCIMGroup(const MCIMGroup &other)
    : accessFunction(other.accessFunction->clone()) {}

MCIMGroup::~MCIMGroup() = default;

const AccessFunction &MCIMGroup::getAccessFunction() const {
  assert(accessFunction != nullptr && "Access function not set");
  return *accessFunction;
}
} // namespace marco::modeling
