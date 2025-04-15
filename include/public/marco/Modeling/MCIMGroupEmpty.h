#include "marco/Modeling/AccessFunctionEmpty.h"
#include "marco/Modeling/MCIMGroupGeneric.h"

#ifndef MARCO_MODELING_MCIMGROUPEMPTY_H
#define MARCO_MODELING_MCIMGROUPEMPTY_H

namespace marco::modeling {
class MCIMGroupEmpty : public MCIMGroupGeneric {
public:
  static std::unique_ptr<MCIMGroup>
  build(const AccessFunctionEmpty &accessFunction);

  explicit MCIMGroupEmpty(const AccessFunction &accessFunction);

  MCIMGroupEmpty(const MCIMGroupEmpty &other);

  ~MCIMGroupEmpty() override;

  std::unique_ptr<MCIMGroup> clone() const override;

  IndexSet getValues() const override;

  bool hasValue(const Point &point) const override;

  void removeValues(const IndexSet &removedValues) override;

  bool has(const Point &key, const Point &value) const override;

  bool set(const Point &key, const Point &value) override;

  bool unset(const Point &key, const Point &value) override;

  std::unique_ptr<MCIMGroup>
  filterValues(const IndexSet &filter) const override;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_MCIMGROUPEMPTY_H
