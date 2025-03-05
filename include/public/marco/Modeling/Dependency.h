#ifndef MARCO_MODELING_DEPENDENCY_H
#define MARCO_MODELING_DEPENDENCY_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/TreeOStream.h"

namespace marco::modeling {
namespace dependency {
// This class must be specialized for the variable type that is used during
// the cycles identification process.
template <typename VariableType>
struct VariableTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the variable.
  //
  // static Id getId(const VariableType*)
  //    return the ID of the variable.

  using Id = typename VariableType::UnknownVariableTypeError;
};

// This class must be specialized for the equation type that is used during
// the cycles identification process.
template <typename EquationType>
struct EquationTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the equation.
  //
  // static Id getId(const EquationType*)
  //    return the ID of the equation.
  //
  // static size_t getNumOfIterationVars(const EquationType*)
  //    return the number of induction variables.
  //
  // static IndexSet getIterationRanges(const EquationType*)
  //    return the iteration ranges.
  //
  // typedef VariableType : the type of the accessed variable
  //
  // typedef AccessProperty : the access property (this is optional, and if not
  // specified an empty one is used)
  //
  // static std::vector<Access<VariableType, AccessProperty>>
  // getWrites(const EquationType*)
  //    return the writes accesses done by the equation.
  //
  // static std::vector<Access<VariableType, AccessProperty>>
  // getReads(const EquationType*)
  //    return the read accesses done by the equation.

  using Id = typename EquationType::UnknownEquationTypeError;
};
} // namespace dependency

namespace internal::dependency {
/// Fallback access property, in case the user didn't provide one.
class EmptyAccessProperty {};

/// Determine the access property to be used according to the user-provided
/// equation property.
template <class T>
struct get_access_property {
  template <typename U>
  using Traits = ::marco::modeling::dependency::EquationTraits<U>;

  template <class U, typename = typename Traits<U>::AccessProperty>
  static typename Traits<U>::AccessProperty property(int);

  template <class U>
  static EmptyAccessProperty property(...);

  using type = decltype(property<T>(0));
};
} // namespace internal::dependency

namespace dependency {
template <typename VariableProperty,
          typename AccessProperty = internal::dependency::EmptyAccessProperty>
class Access {
public:
  using Property = AccessProperty;
  using VariableId = typename VariableTraits<VariableProperty>::Id;

  Access(const VariableProperty &variable,
         std::unique_ptr<AccessFunction> accessFunction,
         AccessProperty property = AccessProperty())
      : variable(VariableTraits<VariableProperty>::getId(&variable)),
        accessFunction(std::move(accessFunction)),
        property(std::move(property)) {}

  Access(const Access &other)
      : variable(other.variable), accessFunction(other.accessFunction->clone()),
        property(other.property) {}

  friend void swap(Access &first, Access &second) {
    using std::swap;
    swap(first.variable, second.variable);
    swap(first.accessFunction, second.accessFunction);
    swap(first.property, second.property);
  }

  Access &operator=(const Access &other) {
    Access result(other);
    swap(*this, result);
    return *this;
  }

  /// Get the ID of the accesses variable.
  VariableId getVariable() const { return variable; }

  /// Get the access function.
  const AccessFunction &getAccessFunction() const { return *accessFunction; }

  /// Get the user-defined access property.
  const AccessProperty &getProperty() const { return property; }

private:
  typename VariableTraits<VariableProperty>::Id variable;
  std::unique_ptr<AccessFunction> accessFunction;
  AccessProperty property;
};
} // namespace dependency

namespace internal::dependency {
/// Wrapper for variables.
template <typename VariableProperty>
class ArrayVariable {
public:
  using Property = VariableProperty;
  using Traits =
      ::marco::modeling::dependency::VariableTraits<VariableProperty>;
  using Id = typename Traits::Id;

  explicit ArrayVariable(VariableProperty property) : property(property) {}

  bool operator==(const ArrayVariable &other) const {
    return getId() == other.getId();
  }

  Id getId() const { return property.getId(); }

  VariableProperty &getProperty() { return property; }

  const VariableProperty &getProperty() const { return property; }

private:
  // Custom variable property.
  VariableProperty property;
};

/// Utility class providing additional methods relying on the ones provided by
/// the user specialization.
template <typename EquationProperty>
class ArrayEquationTraits {
private:
  using Traits =
      ::marco::modeling::dependency::EquationTraits<EquationProperty>;

public:
  using Id = typename Traits::Id;
  using VariableType = typename Traits::VariableType;
  using AccessProperty = typename get_access_property<EquationProperty>::type;

  /// @name Forwarding methods
  /// {

  static Id getId(const EquationProperty *equation) {
    return Traits::getId(equation);
  }

  static size_t getNumOfIterationVars(const EquationProperty *equation) {
    return Traits::getNumOfIterationVars(equation);
  }

  static IndexSet getIterationRanges(const EquationProperty *equation) {
    return Traits::getIterationRanges(equation);
  }

  using Access =
      ::marco::modeling::dependency::Access<VariableType, AccessProperty>;

  static std::vector<Access> getWrites(const EquationProperty *equation) {
    return Traits::getWrites(equation);
  }

  static std::vector<Access> getReads(const EquationProperty *equation) {
    return Traits::getReads(equation);
  }

  /// }
};

/// Wrapper for equations.
template <typename EquationProperty>
class ArrayEquation {
public:
  using Property = EquationProperty;
  using Traits = ArrayEquationTraits<EquationProperty>;
  using Id = typename Traits::Id;
  using Access = typename Traits::Access;

  explicit ArrayEquation(EquationProperty property) : property(property) {}

  bool operator==(const ArrayEquation &other) const {
    return getId() == other.getId();
  }

  EquationProperty &getProperty() { return property; }

  const EquationProperty &getProperty() const { return property; }

  /// @name Forwarding methods
  /// {

  Id getId() const { return Traits::getId(&property); }

  size_t getNumOfIterationVars() const {
    return Traits::getNumOfIterationVars(&property);
  }

  Range getIterationRange(size_t index) const {
    return Traits::getIterationRange(&property, index);
  }

  IndexSet getIterationRanges() const {
    return Traits::getIterationRanges(&property);
  }

  std::vector<Access> getWrites() const { return Traits::getWrites(&property); }

  std::vector<Access> getReads() const { return Traits::getReads(&property); }

  /// }

private:
  // Custom equation property.
  EquationProperty property;
};

/// Keeps track of which variable, together with its indexes, are written
/// by an equation.
template <typename Graph, typename VariableId, typename EquationDescriptor>
class WriteInfo : public Dumpable {
public:
  WriteInfo(const Graph &graph, VariableId variable,
            EquationDescriptor equation, IndexSet indexes)
      : graph(&graph), variable(std::move(variable)),
        equation(std::move(equation)), indexes(std::move(indexes)) {}

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Write information\n";
    os << "Variable: " << variable << "\n";
    os << "Equation: " << (*graph)[equation].getId() << "\n";
    os << "Written variable indexes: " << indexes << "\n";
  }

  const VariableId &getVariable() const { return variable; }

  EquationDescriptor getEquation() const { return equation; }

  const IndexSet &getWrittenVariableIndexes() const { return indexes; }

private:
  // Used for debugging purpose.
  const Graph *graph;
  VariableId variable;

  EquationDescriptor equation;
  IndexSet indexes;
};
} // namespace internal::dependency
} // namespace marco::modeling

#endif // MARCO_MODELING_DEPENDENCY_H
