#ifndef MARCO_MODELING_PANTELIDES_H
#define MARCO_MODELING_PANTELIDES_H

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "pantelides"
#endif

#include "marco/Modeling/Matching.h"

namespace marco::modeling {
namespace internal::pantelides {
namespace singularity {
template <typename EquationProperty>
class EquationWrapper : public Dumpable {
  using EquationTraits = modeling::matching::EquationTraits<EquationProperty>;

public:
  using Id = typename EquationTraits::Id;

private:
  /// The identifier of the equation.
  Id id;

  /// Custom equation property.
  std::shared_ptr<EquationProperty> property;

public:
  explicit EquationWrapper(std::shared_ptr<EquationProperty> property)
      : EquationWrapper(getId(*property), property) {}

private:
  EquationWrapper(Id id, std::shared_ptr<EquationProperty> property)
      : id(std::move(id)), property(std::move(property)) {}

public:
  const Id &getId() const { return id; }

  EquationProperty &getProperty() {
    assert(property && "Property not set");
    return *property;
  }

  const EquationProperty &getProperty() const {
    assert(property && "Property not set");
    return *property;
  }

  void dump(llvm::raw_ostream &os) const override { os << getId() << " @ "; }

private:
  static Id getId(const EquationProperty &p) {
    return EquationTraits::getId(&p);
  }
};
} // namespace singularity
} // namespace internal::pantelides
namespace internal::pantelides {
template <typename VertexDescriptor>
struct VertexSubset {
  VertexDescriptor descriptor;
  IndexSet indices;
};

template <typename VariableProperty, typename EquationProperty>
class PantelidesGraph : public matching::MatchingGraphCRTP<
                            PantelidesGraph<VariableProperty, EquationProperty>,
                            VariableProperty, EquationProperty> {
public:
  using BaseGraph =
      matching::MatchingGraphCRTP<PantelidesGraph, VariableProperty,
                                  EquationProperty>;

  using VertexDescriptor = typename BaseGraph::VertexDescriptor;
  using EdgeDescriptor = typename BaseGraph::EdgeDescriptor;
  using Vertex = typename BaseGraph::Vertex;
  using Edge = typename BaseGraph::Edge;

  using Variable = typename BaseGraph::Variable;
  using Equation = typename BaseGraph::Equation;

  using VariableTraits = typename Variable::Traits;
  using EquationTraits = typename Equation::EquationTraits;

  using VariableId = typename Variable::Id;
  using EquationId = typename Equation::Id;

  using TraversableEdges = typename BaseGraph::TraversableEdges;
  using BFSStep = typename BaseGraph::BFSStep;
  using Frontier = typename BaseGraph::Frontier;
  using MatchingOptions = typename BaseGraph::MatchingOptions;
  using MatchingSolution = typename BaseGraph::MatchingSolution;
  using Visits = typename BaseGraph::Visits;
  using AugmentingPath = typename BaseGraph::AugmentingPath;

  using VariableDifferentiationFn = std::function<VariableProperty(
      const VariableProperty &, const IndexSet &)>;

  using EquationDifferentiationFn = std::function<EquationProperty(
      const EquationProperty &, const IndexSet &)>;

private:
  /// Callback function to create the time derivative of a variable.
  VariableDifferentiationFn variableDifferentiationFn;

  /// Callback function to create the time derivative of an equation.
  EquationDifferentiationFn equationDifferentiationFn;

  using VariableSubset = VertexSubset<VertexDescriptor>;
  using EquationSubset = VertexSubset<VertexDescriptor>;

  /// Associates a variable with its derivative.
  /// v -> (v', indices of v)
  llvm::DenseMap<VertexDescriptor, VariableSubset> variableDerivatives;

  /// Associates an equation with its derivative.
  /// e -> (e', indices of e)
  llvm::DenseMap<VertexDescriptor, EquationSubset> equationDerivatives;

public:
  PantelidesGraph(mlir::MLIRContext *context,
                  VariableDifferentiationFn variableDifferentiationFn,
                  EquationDifferentiationFn equationDifferentiationFn)
      : BaseGraph(context),
        variableDifferentiationFn(std::move(variableDifferentiationFn)),
        equationDifferentiationFn(std::move(equationDifferentiationFn)) {}

  PantelidesGraph(const PantelidesGraph &other) = delete;

  PantelidesGraph(PantelidesGraph &&) noexcept = default;

  PantelidesGraph &operator=(const PantelidesGraph &other) = delete;

  PantelidesGraph &operator=(PantelidesGraph &&) noexcept = default;

  PantelidesGraph newInstance() const override {
    return PantelidesGraph(this->getContext(), variableDifferentiationFn,
                           equationDifferentiationFn);
  }

  void addVariableDerivative(const typename VariableTraits::Id &variableId,
                             const typename VariableTraits::Id &derivativeId,
                             const IndexSet &indices) {
    auto variableDescriptor =
        this->getVariableDescriptorFromId(VariableId(variableId, std::nullopt));

    auto derivativeDescriptor = this->getVariableDescriptorFromId(
        VariableId(derivativeId, std::nullopt));

    if (variableDescriptor && derivativeDescriptor) {
      addVariableDerivative(*variableDescriptor, *derivativeDescriptor,
                            indices);
    }
  }

  void addVariableDerivative(VertexDescriptor variableDescriptor,
                             VertexDescriptor derivativeDescriptor,
                             const IndexSet &indices) {
    Variable &variable = this->getVariableFromDescriptor(variableDescriptor);

    if (variable.removeIndices(indices)) {
      for (EdgeDescriptor edgeDescriptor :
           llvm::make_range(this->edgesBegin(variableDescriptor),
                            this->edgesEnd(variableDescriptor))) {
        Edge &edge = this->getBaseGraph()[edgeDescriptor];
        edge.setVariableIndices(variable.getIndicesPtr());
      }

      auto it = variableDerivatives.find(variableDescriptor);

      if (it == variableDerivatives.end()) {
        LLVM_DEBUG(llvm::dbgs() << "Setting derived indices for variable "
                                << variable.getId() << ": " << indices << "\n");

        variableDerivatives[variableDescriptor] =
            VariableSubset{derivativeDescriptor, std::move(indices)};
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Adding derived indices for variable "
                                << variable.getId() << ": " << indices << "\n");

        it->second.indices += indices;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Removing variable " << variable.getId() << "\n");
      this->removeVariable(variableDescriptor);
    }
  }

private:
  bool hasVariableDerivative(VertexDescriptor variable) const {
    auto it = variableDerivatives.find(variable);
    return it != variableDerivatives.end();
  }

  bool hasVariableDerivative(VertexDescriptor variable,
                             const IndexSet &indices) const {
    auto it = variableDerivatives.find(variable);

    if (it == variableDerivatives.end()) {
      return false;
    }

    return it->second.indices.overlaps(indices);
  }

  const VariableSubset &getVariableDerivative(VertexDescriptor variable) const {
    auto it = variableDerivatives.find(variable);
    assert(it != variableDerivatives.end());
    return it->second;
  }

public:
  void addEquationDerivative(VertexDescriptor equation,
                             VertexDescriptor derivative, IndexSet indices) {
    auto it = equationDerivatives.find(equation);

    if (it == equationDerivatives.end()) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Setting derived indices for equation "
                     << this->getEquationFromDescriptor(equation).getId()
                     << "\n";);

      equationDerivatives[equation] =
          VariableSubset{derivative, std::move(indices)};
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Adding derived indices for equation "
                 << this->getEquationFromDescriptor(equation) << "\n");

      it->second.indices += indices;
    }
  }

private:
  bool hasEquationDerivative(VertexDescriptor equation) const {
    auto it = equationDerivatives.find(equation);
    return it != equationDerivatives.end();
  }

  bool hasEquationDerivative(VertexDescriptor equation,
                             const IndexSet &indices) const {
    auto it = equationDerivatives.find(equation);

    if (it == equationDerivatives.end()) {
      return false;
    }

    return it->second.indices.overlaps(indices);
  }

  const EquationSubset &getEquationDerivative(VertexDescriptor equation) const {
    auto it = equationDerivatives.find(equation);
    assert(it != equationDerivatives.end());
    return it->second;
  }

  /// Check if the system is structurally singular.
  bool isStructurallySingular() const {
    // TODO Check for structural singularity.
    return false;
  }

public:
  bool run() {
    if (isStructurallySingular()) {
      // The Pantelides algorithm would not terminate.
      LLVM_DEBUG(llvm::dbgs() << "The system is structurally singular\n");
      return false;
    }

    llvm::SmallVector<MatchingSolution> matchingSolutions;

    if (this->match(matchingSolutions)) {
      // The system has already index 1, there is no need to run the algorithm.
      return true;
    }

    // Keep track of the original variables and equations.
    llvm::SmallVector<VertexDescriptor> originalVariables;
    llvm::SmallVector<VertexDescriptor> originalEquations;

    for (VertexDescriptor originalVertex : this->getBaseGraph().getVertices()) {
      if (this->isVariable(originalVertex)) {
        originalVariables.push_back(originalVertex);
      } else {
        assert(this->isEquation(originalVertex));
        originalEquations.push_back(originalVertex);
      }
    }

    llvm::SmallVector<VertexDescriptor> equationsWorkList;
    llvm::append_range(equationsWorkList, originalEquations);

    while (!equationsWorkList.empty()) {
      VertexDescriptor equationDescriptor = equationsWorkList.pop_back_val();

      const Equation &equation =
          this->getEquationFromDescriptor(equationDescriptor);

      std::vector<AugmentingPath> augmentingPaths;

      do {
        const IndexSet &unmatchedIndices = equation.getUnmatched();

        if (unmatchedIndices.empty()) {
          // The equation has been fully matched.
          break;
        }

        LLVM_DEBUG(llvm::dbgs() << "Current equation: " << equation.getId()
                                << " @ " << unmatchedIndices << "\n");

        augmentingPaths.clear();

        // Traverse the graph and get candidate augmenting paths together with
        // the visited nodes.
        Frontier frontier;
        frontier.emplace_back(this->getBaseGraph(), equationDescriptor,
                              unmatchedIndices);

        TraversableEdges traversableEdges;
        Visits visits;

        this->collectTraversableEdges(traversableEdges);

        std::vector<std::shared_ptr<BFSStep>> candidateAugmentingPaths =
            this->getCandidateAugmentingPaths(frontier, traversableEdges,
                                              &visits);

        if (!candidateAugmentingPaths.empty()) {
          // TODO should visits come from the filtering?
          this->filterCandidateAugmentingPaths(augmentingPaths,
                                               candidateAugmentingPaths);
        }

        if (!augmentingPaths.empty()) {
          // Apply the augmenting paths.
          for (auto &path : augmentingPaths) {
            this->applyPath(path, traversableEdges);
          }
        } else {
          differentiateVisitedVariables(visits);
          differentiateVisitedEquations(visits);

          equationsWorkList.push_back(
              getEquationDerivative(equationDescriptor).descriptor);
        }
      } while (!augmentingPaths.empty());
    }

    return true;
  }

private:
  void differentiateVisitedVariables(Visits &visits) {
    for (const auto &[variableDescriptor, indices] :
         visits.getVisitedVariables()) {
      differentiateVariable(variableDescriptor, indices);
    }
  }

  void differentiateVariable(VertexDescriptor variableDescriptor,
                             const IndexSet &indices) {
    const Variable &variable =
        this->getVariableFromDescriptor(variableDescriptor);

    LLVM_DEBUG(llvm::dbgs() << "Differentiating variable " << variable.getId()
                            << " @ " << indices << "\n");

    IndexSet indicesToBeDifferentiated = indices;

    if (hasVariableDerivative(variableDescriptor)) {
      const IndexSet &alreadyDifferentiatedIndices =
          getVariableDerivative(variableDescriptor).indices;

      LLVM_DEBUG(llvm::dbgs() << "Found already differentiated indices:"
                              << alreadyDifferentiatedIndices << "\n");

      indicesToBeDifferentiated -= alreadyDifferentiatedIndices;
    }

    if (indicesToBeDifferentiated.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No remaining indices to differentiate\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Differentiating indices "
                            << indicesToBeDifferentiated << "\n");

    VariableProperty differentiatedVariableProperty = variableDifferentiationFn(
        variable.getProperty(), indicesToBeDifferentiated);

    Variable differentiatedVariable(std::make_shared<VariableProperty>(
        std::move(differentiatedVariableProperty)));

    const VariableId &differentiatedVariableId = differentiatedVariable.getId();
    VertexDescriptor differentiatedVariableDescriptor;

    if (this->hasVariableWithId(differentiatedVariableId)) {
      auto optionalDescriptor =
          this->getVariableDescriptorFromId(differentiatedVariableId);

      assert(optionalDescriptor);
      differentiatedVariableDescriptor = *optionalDescriptor;
    } else {
      differentiatedVariableDescriptor =
          this->addVariable(std::move(differentiatedVariable));
    }

    addVariableDerivative(variableDescriptor, differentiatedVariableDescriptor,
                          indicesToBeDifferentiated);
  }

  void differentiateVisitedEquations(Visits &visits) {
    for (const auto &[equationDescriptor, indices] :
         visits.getVisitedEquations()) {
      differentiateEquation(equationDescriptor, indices);
    }
  }

  void differentiateEquation(VertexDescriptor equationDescriptor,
                             const IndexSet &indices) {
    const Equation &equation =
        this->getEquationFromDescriptor(equationDescriptor);

    LLVM_DEBUG(llvm::dbgs() << "Differentiating equation " << equation.getId()
                            << " @ " << indices << "\n");

    IndexSet indicesToBeDifferentiated = indices;

    if (hasEquationDerivative(equationDescriptor)) {
      const IndexSet &alreadyDifferentiatedIndices =
          getEquationDerivative(equationDescriptor).indices;

      LLVM_DEBUG(llvm::dbgs() << "Found already differentiated indices:"
                              << alreadyDifferentiatedIndices << "\n");

      indicesToBeDifferentiated -= alreadyDifferentiatedIndices;
    }

    if (indicesToBeDifferentiated.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No remaining indices to differentiate\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Differentiating indices "
                            << indicesToBeDifferentiated << "\n");

    EquationProperty differentiatedEquationProperty = equationDifferentiationFn(
        equation.getProperty(), indicesToBeDifferentiated);

    Equation differentiatedEquation(
        this->getContext(), std::make_shared<EquationProperty>(
                                std::move(differentiatedEquationProperty)));

    const EquationId &differentiatedEquationId = differentiatedEquation.getId();
    VertexDescriptor differentiatedEquationDescriptor;

    if (this->hasEquationWithId(differentiatedEquationId)) {
      auto optionalDescriptor =
          this->getEquationDescriptorFromId(differentiatedEquationId);

      assert(optionalDescriptor);
      differentiatedEquationDescriptor = *optionalDescriptor;
    } else {
      differentiatedEquationDescriptor =
          this->addEquation(std::move(differentiatedEquation));

      this->discoverAccesses(differentiatedEquationDescriptor);
    }

    addEquationDerivative(equationDescriptor, differentiatedEquationDescriptor,
                          indicesToBeDifferentiated);
  }
};
} // namespace internal::pantelides
} // namespace marco::modeling

#endif // MARCO_MODELING_PANTELIDES_H
