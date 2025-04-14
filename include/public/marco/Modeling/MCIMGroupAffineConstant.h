#include "marco/Modeling/AccessFunctionAffineConstant.h"
#include "marco/Modeling/MCIMGroupGeneric.h"

#ifndef MARCO_MODELING_MCIMGROUPAFFINECONSTANT_H
#define MARCO_MODELING_MCIMGROUPAFFINECONSTANT_H

namespace marco::modeling {
class MCIMGroupAffineConstant : public MCIMGroupGeneric {
  IndexSet mapped;

public:
  static std::unique_ptr<MCIMGroup>
  build(const AccessFunctionAffineConstant &accessFunction);

  explicit MCIMGroupAffineConstant(
      const AccessFunctionAffineConstant &accessFunction);

  MCIMGroupAffineConstant(const MCIMGroupAffineConstant &other);

  ~MCIMGroupAffineConstant() override;

  std::unique_ptr<MCIMGroup> clone() const override;

  bool hasValue(const Point &point) const override;

  void removeValues(const IndexSet &removedValues) override;

  bool has(const Point &key, const Point &value) const override;

  bool set(const Point &key, const Point &value) override;

  bool unset(const Point &key, const Point &value) override;

  std::unique_ptr<MCIMGroup>
  filterValues(const IndexSet &filter) const override;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_MCIMGROUPAFFINECONSTANT_H
