#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANTIATOR_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANTIATOR_H

#include "marco/Dialect/Modelica/Transforms/Instantiation/InstanceTree.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include <deque>
#include <memory>

namespace mlir::modelica::instantiation {
/// Values of the indices of the enclosing arrays of instances, by position.
/// Null values represent unknown indices.
using IndexValues = llvm::SmallVector<mlir::Value>;

/// Component referenced by an SSA value obtained from a component reference.
struct ReferencedComponent {
  Component *component;

  /// Values of the indices of the arrays of instances along the path, by
  /// position. Null values represent indices that are not subscripted.
  IndexValues indices;
};

/// Instantiation of a class. The class-level operations of
/// a module are instantiated into a 'modelica.model', together with the
/// functions it uses.
class Instantiator {
public:
  explicit Instantiator(mlir::ModuleOp module);

  ~Instantiator();

  /// Instantiate the class with the given dotted name.
  mlir::LogicalResult instantiateModel(llvm::StringRef name);

  /// Erase the class-level operations from the module.
  void eraseClasses();

  //===-------------------------------------------------------------------===//
  // Scopes and lookup
  //===-------------------------------------------------------------------===//

  Scope *createScope(mlir::Operation *op, Scope *parent, Modifier modifier,
                     Component *instance, Element *replaced);

  Scope *createPredefinedScope(PredefinedClass predefined, Modifier modifier,
                               Component *instance);

  /// Create the local elements of a class.
  mlir::LogicalResult createLocals(Scope *scope);

  /// Expand the class, adding the inherited elements.
  mlir::LogicalResult expand(Scope *scope);

  /// Look up a simple name starting from the given scope.
  /// If 'skipInherited' is set, the elements inherited
  /// by the starting scope are not considered, as done for the names of the
  /// base classes.
  Element *lookupSimple(Scope *scope, llvm::StringRef name,
                        bool skipInherited = false);

  /// Look up a name, possibly composite or global.
  Element *lookupName(Scope *scope, mlir::SymbolRefAttr name,
                      bool skipInherited = false);

  /// Look up an element of a class, including the inherited ones.
  Element *lookupMember(Scope *scope, llvm::StringRef name);

  /// Get the scope of a class element, without modifications.
  Scope *getClassScope(Element *element);

  /// Instantiate a class element with the given modification.
  Scope *instantiateClass(Element *element, Modifier modifier,
                          Component *instance);

  /// Get the absolute symbol path of a class.
  mlir::SymbolRefAttr getClassSymbol(Scope *scope);

  //===-------------------------------------------------------------------===//
  // Components
  //===-------------------------------------------------------------------===//

  /// Get the component of a component element, instantiating its class.
  Component *getComponent(Element *element);

  /// Get the dimensions of a component, evaluating them if needed.
  mlir::FailureOr<llvm::ArrayRef<int64_t>> getDims(Component *component);

  /// Get whether a conditional component is enabled.
  mlir::FailureOr<bool> isEnabled(Component *component);

  /// Get the inner component corresponding to an outer one.
  Component *resolveOuter(Component *component);

  /// Get the type of the values of a component.
  mlir::FailureOr<mlir::Type> getType(Component *component);

  /// Get the type of the scalar values of a primitive class.
  mlir::Type getScalarType(Scope *primitive);

  //===-------------------------------------------------------------------===//
  // Expressions
  //===-------------------------------------------------------------------===//

  /// Emit a class-level value, together with the operations it depends on,
  /// resolving the names it uses in the given scope.
  mlir::FailureOr<mlir::Value> emitValue(mlir::Value value, Scope *scope,
                                         const IndexValues &indices,
                                         mlir::OpBuilder &builder,
                                         mlir::IRMapping &mapping);

  /// Clone a class-level operation, resolving the names it uses in the
  /// given scope.
  mlir::FailureOr<mlir::Operation *> emitOp(mlir::Operation *op, Scope *scope,
                                            const IndexValues &indices,
                                            mlir::OpBuilder &builder,
                                            mlir::IRMapping &mapping);

  /// Emit the value of a modification, applying the indices of the arrays
  /// of instances it is split along.
  mlir::FailureOr<mlir::Value> emitModifierValue(const ModifierValue &value,
                                                 const IndexValues &indices,
                                                 mlir::OpBuilder &builder);

  /// Emit an absolute reference to a component.
  mlir::FailureOr<mlir::Value> emitReference(Component *component,
                                             const IndexValues &indices,
                                             mlir::OpBuilder &builder,
                                             mlir::Location loc);

  /// Get the component referenced by an emitted value, if any.
  const ReferencedComponent *getReferencedComponent(mlir::Value value) const;

  //===-------------------------------------------------------------------===//
  // Evaluation
  //===-------------------------------------------------------------------===//

  /// Evaluate a class-level value in the given scope. The indices of the
  /// enclosing arrays of instances are unknown.
  mlir::Attribute evaluate(mlir::Value value, Scope *scope,
                           unsigned numIndices,
                           const mlir::IRMapping &mapping = {});

  /// Evaluate a class-level value in the given scope, given the values of
  /// the indices of the enclosing arrays of instances.
  mlir::Attribute evaluate(mlir::Value value, Scope *scope,
                           llvm::ArrayRef<std::optional<int64_t>> indices,
                           const mlir::IRMapping &mapping = {});

  /// Evaluate a structural expression of a component, that is an expression
  /// that must have the same value for all the elements of the enclosing
  /// arrays of instances. The expression is evaluated for each element if
  /// it depends on their indices.
  mlir::FailureOr<mlir::Attribute>
  evaluateStructural(mlir::Value value, Scope *scope, Component *component,
                     mlir::Location loc);

  /// Evaluate the value of a component, given the values of the indices
  /// of the enclosing arrays of instances.
  mlir::Attribute
  evaluateComponent(Component *component,
                    llvm::ArrayRef<std::optional<int64_t>> indices);

  /// Evaluate an emitted value.
  mlir::Attribute evaluateEmitted(mlir::Value value);

  //===-------------------------------------------------------------------===//
  // Functions
  //===-------------------------------------------------------------------===//

  /// Get the instantiated function corresponding to a class element.
  mlir::FailureOr<mlir::FlatSymbolRefAttr> getFunction(Element *element,
                                                       mlir::Location loc);

private:
  /// Temporary container for the operations emitted during evaluations,
  /// erased on destruction.
  struct ScratchArea {
    Instantiator *instantiator;
    ModelOp container;
    mlir::OpBuilder builder;

    ScratchArea(Instantiator *instantiator, ModelOp container);

    ~ScratchArea();
  };

  std::unique_ptr<ScratchArea> createScratchArea();

  /// Resolve the names used by an emitted operation and by the operations
  /// nested within it, and update the types of the results. If the root
  /// operation is replaced, the replacement values are returned.
  mlir::LogicalResult fixup(mlir::Operation *root, Scope *scope,
                            const IndexValues &indices,
                            llvm::SmallVectorImpl<mlir::Value> &replacement);

  /// The following functions return the value replacing the operation, if
  /// any, or a null value.
  mlir::FailureOr<mlir::Value> fixupReference(RefOp op, Scope *scope,
                                              const IndexValues &indices);

  mlir::FailureOr<mlir::Value> fixupClassReference(RefOp op, Scope *scope);

  mlir::FailureOr<mlir::Value> fixupMember(MemberOp op);

  mlir::LogicalResult fixupSubscript(SubscriptOp op);

  mlir::FailureOr<mlir::Value> fixupCall(mlir::Operation *op, Scope *scope);

  /// Get the placeholder of an unknown index, which can only be used within
  /// evaluations.
  mlir::FailureOr<mlir::Value> getUnknownIndex(mlir::OpBuilder &builder,
                                               mlir::Location loc);

  /// Emit the value of a constant of a package.
  mlir::FailureOr<mlir::Value> emitPackageConstant(Component *component,
                                                   mlir::OpBuilder &builder,
                                                   mlir::Location loc);

  mlir::LogicalResult instantiateComponent(Component *component);

  mlir::FailureOr<int64_t> evaluateDimension(DimensionAttr dimension,
                                             mlir::Value value, Scope *scope,
                                             Component *component,
                                             mlir::Location loc);

  //===-------------------------------------------------------------------===//
  // Models
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult emitMembers(Component *instance, mlir::OpBuilder &builder);

  mlir::LogicalResult emitMember(Component *component, mlir::OpBuilder &builder);

  mlir::LogicalResult emitVariableModification(Component *component,
                                               VariableOp variableOp);

  mlir::LogicalResult emitSections(Component *instance,
                                   mlir::Operation *container);

  mlir::LogicalResult emitEquation(mlir::Operation *op, Scope *scope,
                                   const IndexValues &indices,
                                   mlir::OpBuilder &builder,
                                   mlir::IRMapping &mapping);

  mlir::LogicalResult emitConnect(ConnectOp op, Scope *scope,
                                  const IndexValues &indices,
                                  mlir::OpBuilder &builder,
                                  mlir::IRMapping &mapping);

  mlir::LogicalResult emitForEquation(ForEquationOp op, Scope *scope,
                                      const IndexValues &indices,
                                      mlir::OpBuilder &builder,
                                      mlir::IRMapping &mapping);

  mlir::LogicalResult emitIfEquation(IfEquationOp op, Scope *scope,
                                     const IndexValues &indices,
                                     mlir::OpBuilder &builder,
                                     mlir::IRMapping &mapping);

  /// Get the sizes of the dimensions of the arrays of instances enclosing a
  /// component, and of the component itself.
  mlir::FailureOr<llvm::SmallVector<int64_t>>
  getIndexSizes(Component *component);

private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;

  std::deque<std::unique_ptr<Scope>> scopes;
  std::deque<std::unique_ptr<Element>> elements;
  std::deque<std::unique_ptr<Component>> components;

  Scope *rootScope = nullptr;
  llvm::DenseMap<Element *, Scope *> predefinedScopes;

  /// Class operations of the module before instantiation.
  llvm::SmallVector<mlir::Operation *> classOps;

  /// Components referenced by the emitted values.
  llvm::DenseMap<mlir::Value, ReferencedComponent> referencedComponents;

  /// Evaluations of components, by component and indices.
  std::map<std::pair<Component *, std::vector<int64_t>>, mlir::Attribute>
      evaluations;
  llvm::DenseSet<Component *> activeEvaluations;
  llvm::DenseSet<Component *> activeInstantiations;
  llvm::DenseSet<Component *> activeDimensions;

  /// Instantiated functions, by class scope.
  llvm::DenseMap<Scope *, mlir::FlatSymbolRefAttr> functions;
  llvm::StringSet<> functionNames;

  /// Scopes identifying the functions declared within unmodified packages.
  llvm::DenseMap<mlir::Operation *, Scope *> canonicalFunctionScopes;

  /// The model being instantiated.
  ModelOp modelOp;
  Component *modelRoot = nullptr;
  std::string modelName;

  /// Placeholder for unknown indices.
  mlir::OwningOpRef<mlir::ModuleOp> scratchModule;
  mlir::Value unknownIndex;
};
} // namespace mlir::modelica::instantiation

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_INSTANTIATION_INSTANTIATOR_H
