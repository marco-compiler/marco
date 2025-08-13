#ifndef MARCO_MODELING_SCHEDULING_H
#define MARCO_MODELING_SCHEDULING_H

#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/ArrayEquationsDependencyGraph.h"
#include "marco/Modeling/SCCsDependencyGraph.h"
#include "marco/Modeling/ScalarEquationsDependencyGraph.h"
#include <numeric>

namespace marco::modeling {
namespace internal::scheduling {
template <typename VariableProperty, typename EquationProperty>
class EquationView {
public:
  using EquationTraits =
      typename ::marco::modeling::dependency::EquationTraits<EquationProperty>;

  using Id = typename EquationTraits::Id;
  using VariableType = typename EquationTraits::VariableType;
  using AccessProperty = typename EquationTraits::AccessProperty;

  using Access =
      ::marco::modeling::dependency::Access<VariableType, AccessProperty>;

  EquationView(EquationProperty property, IndexSet indices)
      : property(std::move(property)), indices(std::move(indices)) {}

  EquationProperty &operator*() { return property; }

  const EquationProperty &operator*() const { return property; }

  Id getId() const {
    // Forward the request to the traits class of the property.
    return EquationTraits::getId(&property);
  }

  size_t getNumOfIterationVars() const {
    // Forward the request to the traits class of the property.
    return EquationTraits::getNumOfIterationVars(&property);
  }

  const IndexSet &getIterationRanges() const { return indices; }

  std::vector<Access> getWrites() const {
    // Forward the request to the traits class of the property.
    return EquationTraits::getWrites(&property);
  }

  std::vector<Access> getReads() const {
    // Forward the request to the traits class of the property.
    return EquationTraits::getReads(&property);
  }

private:
  EquationProperty property;
  IndexSet indices;
};

enum class DirectionPossibility { Any, Forward, Backward, Scalar };
} // namespace internal::scheduling

namespace dependency {
template <typename VP, typename EP>
struct EquationTraits<internal::scheduling::EquationView<VP, EP>> {
  using EquationType = internal::scheduling::EquationView<VP, EP>;
  using Id = typename EquationTraits<EP>::Id;

  static Id getId(const EquationType *equation) { return (*equation).getId(); }

  static size_t getNumOfIterationVars(const EquationType *equation) {
    return (*equation).getNumOfIterationVars();
  }

  static IndexSet getIterationRanges(const EquationType *equation) {
    return (*equation).getIterationRanges();
  }

  using VariableType = typename EquationTraits<EP>::VariableType;
  using AccessProperty = typename EquationTraits<EP>::AccessProperty;

  static std::vector<Access<VariableType, AccessProperty>>
  getWrites(const EquationType *equation) {
    return (*equation).getWrites();
  }

  static std::vector<Access<VariableType, AccessProperty>>
  getReads(const EquationType *equation) {
    return (*equation).getReads();
  }
};
} // namespace dependency

namespace scheduling {
/// The direction to be used by the equations loop to update its iteration
/// variable.
enum class Direction { Any, Forward, Backward, Unknown };

template <typename EquationProperty>
class ScheduledEquation {
public:
  ScheduledEquation(EquationProperty property, IndexSet indices,
                    llvm::ArrayRef<Direction> directions)
      : property(std::move(property)), indices(std::move(indices)),
        directions(directions.begin(), directions.end()) {}

  const EquationProperty &getEquation() const { return property; }

  const IndexSet &getIndexes() const { return indices; }

  llvm::ArrayRef<Direction> getIterationDirections() const {
    return directions;
  }

private:
  EquationProperty property;
  IndexSet indices;
  llvm::SmallVector<Direction> directions;
};

template <typename ElementType>
class ScheduledSCC {
private:
  using Container = std::vector<ElementType>;

public:
  using const_iterator = typename Container::const_iterator;

  ScheduledSCC(llvm::ArrayRef<ElementType> equations)
      : equations(equations.begin(), equations.end()) {
    assert(!this->equations.empty());
  }

  const ElementType &operator[](size_t index) const {
    assert(index < equations.size());
    return equations[index];
  }

  size_t size() const { return equations.size(); }

  const_iterator begin() const { return equations.begin(); }

  const_iterator end() const { return equations.end(); }

private:
  Container equations;
};
} // namespace scheduling

/// The scheduler allows to sort the equations in such a way that each scalar
/// variable is determined before being accessed.
/// The scheduling algorithm assumes that all the algebraic loops have
/// already been resolved and the only possible kind of loop is the one given
/// by an equation depending on itself (for example, x[i] = f(x[i - 1]), with
/// i belonging to a range wider than one).
template <typename VariableProperty, typename SCCProperty>
class Scheduler {
private:
  using SCCsGraph = SCCsDependencyGraph<SCCProperty>;
  using SCCTraits = typename SCCsGraph::SCCTraits;
  using EquationDescriptor = typename SCCTraits::ElementRef;
  using EquationTraits = dependency::EquationTraits<EquationDescriptor>;
  using AccessProperty = typename EquationTraits::AccessProperty;

  using Access =
      ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

  using SCCDescriptor = typename SCCsGraph::SCCDescriptor;

  using EquationView =
      internal::scheduling::EquationView<VariableProperty, EquationDescriptor>;

  using ScalarDependencyGraph =
      ScalarEquationsDependencyGraph<VariableProperty, EquationView>;

  using ScheduledEquation = scheduling::ScheduledEquation<EquationDescriptor>;

  using ScheduledSCC = scheduling::ScheduledSCC<ScheduledEquation>;

  using ScheduledSCCs = llvm::SmallVector<ScheduledSCC>;
  using DirectionPossibility = internal::scheduling::DirectionPossibility;

private:
  mlir::MLIRContext *context;

public:
  explicit Scheduler(mlir::MLIRContext *context) : context(context) {}

  [[nodiscard]] mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  ScheduledSCCs schedule(llvm::ArrayRef<SCCProperty> SCCs) const {
    ScheduledSCCs result;

    SCCsGraph SCCsDependencyGraph;
    SCCsDependencyGraph.addSCCs(SCCs);

    for (SCCDescriptor sccDescriptor : SCCsDependencyGraph.postOrder()) {
      const SCCProperty &scc = SCCsDependencyGraph[sccDescriptor];
      auto sccElements = SCCTraits::getElements(&scc);

      if (sccElements.size() == 1) {
        const auto &equation = sccElements[0];
        llvm::SmallVector<DirectionPossibility, 3> directionPossibilities;
        getSchedulingDirections(scc, directionPossibilities);

        assert(directionPossibilities.size() ==
               EquationTraits::getNumOfIterationVars(&equation));

        bool isSchedulableAsRange = llvm::all_of(
            directionPossibilities, [](DirectionPossibility direction) {
              return direction == DirectionPossibility::Any ||
                     direction == DirectionPossibility::Forward ||
                     direction == DirectionPossibility::Backward;
            });

        if (isSchedulableAsRange) {
          llvm::SmallVector<scheduling::Direction> directions;

          for (DirectionPossibility direction : directionPossibilities) {
            if (direction == DirectionPossibility::Any) {
              directions.push_back(scheduling::Direction::Any);
            } else if (direction == DirectionPossibility::Forward) {
              directions.push_back(scheduling::Direction::Forward);
            } else if (direction == DirectionPossibility::Backward) {
              directions.push_back(scheduling::Direction::Backward);
            } else {
              directions.push_back(scheduling::Direction::Unknown);
            }
          }

          ScheduledEquation scheduledEquation(
              equation, EquationTraits::getIterationRanges(&equation),
              directions);

          result.emplace_back(std::move(scheduledEquation));
          continue;
        } else {
          // Mixed accesses detected. Scheduling is possible only on the
          // scalar equations.
          std::vector<EquationView> scalarEquationViews;

          for (const auto &equation : sccElements) {
            scalarEquationViews.push_back(EquationView(
                equation, EquationTraits::getIterationRanges(&equation)));
          }

          ScalarDependencyGraph scalarDependencyGraph(getContext());
          scalarDependencyGraph.addEquations(scalarEquationViews);

          for (const auto &equationDescriptor :
               scalarDependencyGraph.postOrder()) {
            const auto &scalarEquation =
                scalarDependencyGraph[equationDescriptor];

            llvm::SmallVector<scheduling::Direction> directions(
                scalarEquation.getProperty().getNumOfIterationVars(),
                scheduling::Direction::Any);

            ScheduledEquation scheduledEquation(
                *scalarEquation.getProperty(),
                IndexSet(MultidimensionalRange(scalarEquation.getIndex())),
                directions);

            result.emplace_back(std::move(scheduledEquation));
          }
        }
      } else {
        // A strongly connected component with more than one equation can
        // be scheduled with respect to other SCCs, but the equations
        // composing it are cyclic and thus can't be ordered.
        llvm::SmallVector<ScheduledEquation> SCC;

        for (const auto &equation : sccElements) {
          llvm::SmallVector<scheduling::Direction> directions(
              EquationTraits::getNumOfIterationVars(&equation),
              scheduling::Direction::Unknown);

          SCC.push_back(ScheduledEquation(
              equation, EquationTraits::getIterationRanges(&equation),
              directions));
        }

        result.emplace_back(std::move(SCC));
      }
    }

    return result;
  }

private:
  /// Given a SSC containing only one equation that may depend on itself,
  /// determine the access direction with respect to the variable that is
  /// written by the equation.
  void getSchedulingDirections(
      const SCCProperty &scc,
      llvm::SmallVectorImpl<DirectionPossibility> &directions) const {
    auto sccElements = SCCTraits::getElements(&scc);
    assert(sccElements.size() == 1);

    const auto &equation = sccElements[0];
    size_t rank = EquationTraits::getNumOfIterationVars(&equation);

    auto writeAccesses = EquationTraits::getWrites(&equation);

    if (writeAccesses.empty()) {
      directions.append(rank, DirectionPossibility::Any);
      return;
    }

    // Prefer invertible write accesses.
    const auto &writeAccess =
        getAccessWithProperty(writeAccesses, [](const Access &access) {
          return access.getAccessFunction().isInvertible();
        });

    IndexSet equationIndices = EquationTraits::getIterationRanges(&equation);

    auto writtenVariable = writeAccess.getVariable();
    const AccessFunction &writeAccessFunction = writeAccess.getAccessFunction();
    IndexSet writtenIndices(writeAccessFunction.map(equationIndices));

    llvm::SmallVector<std::unique_ptr<AccessFunction>>
        overlappingReadAccessFunctions;

    for (auto &readAccess : EquationTraits::getReads(&equation)) {
      // The access is considered only if it reads the same variable it is
      // being defined by the equation and the ranges overlap.
      const auto &readVariable = readAccess.getVariable();

      if (writtenVariable != readVariable) {
        continue;
      }

      const AccessFunction &readAccessFunction = readAccess.getAccessFunction();
      IndexSet readIndices(readAccessFunction.map(equationIndices));

      if (readIndices.overlaps(writtenIndices)) {
        overlappingReadAccessFunctions.push_back(readAccessFunction.clone());
      }
    }

    if (overlappingReadAccessFunctions.empty()) {
      // If there is no self-loop, then the iteration direction of the equation
      // is irrelevant.
      directions.append(rank, DirectionPossibility::Any);
      return;
    }

    return getSchedulingDirections(directions, writeAccessFunction,
                                   overlappingReadAccessFunctions);
  }

  const Access &getAccessWithProperty(
      llvm::ArrayRef<Access> accesses,
      std::function<bool(const Access &)> preferenceFn) const {
    assert(!accesses.empty());
    auto it = llvm::find_if(accesses, preferenceFn);

    if (it == accesses.end()) {
      it = accesses.begin();
    }

    return *it;
  }

  void getSchedulingDirections(
      llvm::SmallVectorImpl<DirectionPossibility> &directions,
      const AccessFunction &writeAccess,
      llvm::ArrayRef<std::unique_ptr<AccessFunction>> readAccesses) const {
    if (auto rotoTranslationWriteAccess =
            writeAccess.dyn_cast<AccessFunctionRotoTranslation>()) {
      if (getSchedulingDirections(directions, *rotoTranslationWriteAccess,
                                  readAccesses)) {
        return;
      }
    }

    directions.clear();
    directions.resize(writeAccess.getNumOfDims(), DirectionPossibility::Scalar);
  }

  bool getSchedulingDirections(
      llvm::SmallVectorImpl<DirectionPossibility> &directions,
      const AccessFunctionRotoTranslation &writeAccess,
      llvm::ArrayRef<std::unique_ptr<AccessFunction>> readAccesses) const {
    auto inverseWriteAccess = writeAccess.inverse();

    if (!inverseWriteAccess) {
      return false;
    }

    auto inverseRotoTranslationWriteAccess =
        inverseWriteAccess->dyn_cast<AccessFunctionRotoTranslation>();

    if (!inverseRotoTranslationWriteAccess) {
      return false;
    }

    for (const auto &readAccess : readAccesses) {
      llvm::SmallVector<DirectionPossibility, 3> accessDirections;
      auto relativeAccess = inverseWriteAccess->combine(*readAccess);

      auto rotoTranslationRelativeAccess =
          relativeAccess->dyn_cast<AccessFunctionRotoTranslation>();

      if (!rotoTranslationRelativeAccess) {
        return false;
      }

      if (!rotoTranslationRelativeAccess->isIdentityLike()) {
        // If the iteration indices are out of order, then some accesses will
        // refer to future written variables and others to past written ones.
        return false;
      }

      // Analyze the offset of the individual dimension access.
      for (size_t i = 0, e = rotoTranslationRelativeAccess->getNumOfResults();
           i < e; ++i) {
        auto offset = rotoTranslationRelativeAccess->getOffset(i);

        if (offset == 0) {
          accessDirections.push_back(DirectionPossibility::Any);
        } else if (offset > 0) {
          accessDirections.push_back(DirectionPossibility::Backward);
        } else {
          accessDirections.push_back(DirectionPossibility::Forward);
        }
      }

      // Rotate the access directions to match the original dimensions.
      llvm::SmallVector<DirectionPossibility, 3> rotatedAccessDirections;
      rotatedAccessDirections.resize(writeAccess.getNumOfDims(),
                                     DirectionPossibility::Scalar);

      for (size_t i = 0,
                  e = inverseRotoTranslationWriteAccess->getNumOfResults();
           i < e; ++i) {
        auto dimensionIndex =
            inverseRotoTranslationWriteAccess->getInductionVariableIndex(i);

        if (!dimensionIndex) {
          return false;
        }

        rotatedAccessDirections[i] = accessDirections[*dimensionIndex];
      }

      if (directions.empty()) {
        llvm::append_range(directions, rotatedAccessDirections);
      } else {
        mergeDirectionPossibilities(directions, rotatedAccessDirections);
      }

      // If all the inductions have been scalarized, then the situation
      // can't get better anymore.
      if (llvm::all_of(directions, [](DirectionPossibility direction) {
            return direction == DirectionPossibility::Scalar;
          })) {
        return false;
      }
    }

    return true;
  }

  void mergeDirectionPossibilities(
      llvm::SmallVectorImpl<DirectionPossibility> &result,
      llvm::ArrayRef<DirectionPossibility> newDirections) const {
    assert(result.size() == newDirections.size());

    for (size_t i = 0, e = result.size(); i < e; ++i) {
      result[i] = mergeDirectionPossibilities(result[i], newDirections[i]);
    }
  }

  DirectionPossibility
  mergeDirectionPossibilities(DirectionPossibility lhs,
                              DirectionPossibility rhs) const {
    if (lhs == DirectionPossibility::Scalar ||
        rhs == DirectionPossibility::Scalar) {
      return DirectionPossibility::Scalar;
    }

    if (lhs == DirectionPossibility::Any) {
      return rhs;
    }

    if (rhs == DirectionPossibility::Any) {
      return lhs;
    }

    if (lhs != rhs) {
      return DirectionPossibility::Scalar;
    }

    assert(lhs == rhs);
    return lhs;
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_SCHEDULING_H
