#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/MCIMGroup.h"

#ifndef MARCO_MODELING_MCIMGROUPGENERIC_H
#define MARCO_MODELING_MCIMGROUPGENERIC_H

namespace marco::modeling {
class MCIMGroupGeneric : public MCIMGroup {
protected:
  IndexSet keys;

public:
  static std::unique_ptr<MCIMGroup> build(const AccessFunction &accessFunction);

  explicit MCIMGroupGeneric(const AccessFunction &accessFunction);

  MCIMGroupGeneric(const MCIMGroupGeneric &other);

  ~MCIMGroupGeneric() override;

  std::unique_ptr<MCIMGroup> clone() const override;

  bool empty() const override;

  IndexSet getKeys() const override;

  bool hasKey(const Point &key) const override;

  void setKeys(IndexSet newKeys) override;

  void addKeys(const IndexSet &newKeys) override;

  void removeKeys(const IndexSet &removedKeys) override;

  IndexSet getValues() const override;

  bool hasValue(const Point &point) const override;

  void removeValues(const IndexSet &removedValues) override;

  bool has(const Point &key, const Point &value) const override;

  bool set(const Point &key, const Point &value) override;

  bool unset(const Point &key, const Point &value) override;

  std::unique_ptr<MCIMGroup> filterKeys(const IndexSet &filter) const override;

  std::unique_ptr<MCIMGroup>
  filterValues(const IndexSet &filter) const override;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_MCIMGROUPGENERIC_H
