#ifndef MARCO_MODELING_MCIMGROUP_H
#define MARCO_MODELING_MCIMGROUP_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/IndexSet.h"
#include <memory>

namespace marco::modeling {
class MCIMGroup {
public:
  static std::unique_ptr<MCIMGroup> build(const AccessFunction &accessFunction);

  explicit MCIMGroup(const AccessFunction &accessFunction);

  MCIMGroup(const MCIMGroup &other);

  virtual ~MCIMGroup();

  [[nodiscard]] virtual std::unique_ptr<MCIMGroup> clone() const = 0;

  const AccessFunction &getAccessFunction() const;

  virtual bool empty() const = 0;

  virtual IndexSet getKeys() const = 0;

  virtual bool hasKey(const Point &key) const = 0;

  virtual void setKeys(IndexSet newKeys) = 0;

  virtual void addKeys(const IndexSet &newKeys) = 0;

  virtual void removeKeys(const IndexSet &removedKeys) = 0;

  virtual IndexSet getValues() const = 0;

  virtual bool hasValue(const Point &point) const = 0;

  virtual void removeValues(const IndexSet &removedValues) = 0;

  /// Check if a (key, value) pair belongs to the group.
  virtual bool has(const Point &key, const Point &value) const = 0;

  /// Check if a (key, value) pair is allowed in group, and if so, add it.
  virtual bool set(const Point &key, const Point &value) = 0;

  /// Check if a (key, value) pair is allowed in group, and if so, remove it.
  virtual bool unset(const Point &key, const Point &value) = 0;

  /// Get a new group with only the keys, and the associated values, that are
  /// also part of the filter.
  virtual std::unique_ptr<MCIMGroup>
  filterKeys(const IndexSet &filter) const = 0;

  /// Get a new group with only the values, and the associated keys, that are
  /// also part of the filter.
  virtual std::unique_ptr<MCIMGroup>
  filterValues(const IndexSet &filter) const = 0;

private:
  std::unique_ptr<AccessFunction> accessFunction;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_MCIMGROUP_H
