#ifndef MARCO_CODEGEN_LOWERING_MODELICA_LOWERERIMPL_H
#define MARCO_CODEGEN_LOWERING_MODELICA_LOWERERIMPL_H

#include "marco/Codegen/Lowering/Modelica/ClassTree.h"
#include "marco/AST/Modelica/AST.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/TypeUtilities.h"

namespace marco::codegen::lowering::modelica {
/// Implementation of the lowering of the Modelica AST to the class-level
/// operations of the Modelica dialect.
class LowererImpl {
public:
  explicit LowererImpl(mlir::MLIRContext &context);

  ClassTree &getClassTree() { return classTree; }

  /// Lower all the classes of the class tree into the module.
  mlir::LogicalResult lower(mlir::ModuleOp module);

  //===-------------------------------------------------------------------===//
  // Utilities (Lowerer.cpp)
  //===-------------------------------------------------------------------===//

  mlir::Location loc(const ast::ASTNode &node);
  mlir::Location loc(const SourceRange &range);

  /// Emit an error and return failure.
  mlir::LogicalResult emitError(mlir::Location location,
                                const llvm::Twine &message);

  /// Get the symbol reference corresponding to a name as written in the
  /// source code. Global names are rooted at ".".
  mlir::SymbolRefAttr getSymbolRef(const ast::modelica::Name &name);

  /// Get the absolute path of a class.
  mlir::SymbolRefAttr getAbsoluteSymbolRef(ClassNode *classNode);

  /// Convert an annotation into a dictionary attribute.
  mlir::DictionaryAttr
  getAnnotation(const ast::modelica::ClassModification *annotation);

  //===-------------------------------------------------------------------===//
  // Types (Types.cpp)
  //===-------------------------------------------------------------------===//

  /// Get the type of the values of a class.
  mlir::FailureOr<mlir::Type> getClassType(ClassNode *classNode,
                                           mlir::Location location);

  /// Get the type of the values of a component.
  mlir::FailureOr<mlir::Type> getComponentType(const Element &component);

  /// Get the class of the elements of an object type.
  ClassNode *getObjectClass(mlir::Type type);

  /// Get the size of an array dimension, as determined statically.
  int64_t getStaticDimension(const ast::modelica::Expression &subscript,
                             ClassNode *scope);

  /// Get the components of a class in declaration order, including the
  /// inherited ones.
  void collectComponents(ClassNode *classNode,
                         llvm::SmallVectorImpl<Element> &components);

  /// Get the types of the outputs of a function.
  mlir::FailureOr<llvm::SmallVector<mlir::Type>>
  getFunctionResultTypes(ClassNode *function, mlir::Location location);

  //===-------------------------------------------------------------------===//
  // Classes and elements (Lowerer.cpp)
  //===-------------------------------------------------------------------===//

  /// Lower a class of the class tree. Failures are reported, and the class
  /// is omitted from the output.
  void lowerClass(ClassNode *classNode);

  /// Lower a class definition. The node is null for classes declared within
  /// modifications, in which case the scope is the one enclosing the
  /// modification.
  mlir::LogicalResult
  lowerClassDefinition(const ast::modelica::ClassDefinition &definition,
                       ClassNode *classNode, ClassNode *scope,
                       const ast::modelica::ElementPrefixes &prefixes,
                       bool isEach,
                       const ast::modelica::ConstrainingClause *constraint);

  mlir::LogicalResult lowerElement(const ast::modelica::Node &element,
                                   ClassNode *classNode);

  mlir::LogicalResult
  lowerComponent(const ast::modelica::ComponentClause &clause,
                 const ast::modelica::ComponentDeclaration &declaration,
                 const ast::modelica::ElementPrefixes &prefixes, bool isEach,
                 const Element &element,
                 const ast::modelica::ConstrainingClause *constraint);

  mlir::LogicalResult
  lowerExtendsClause(const ast::modelica::ExtendsClause &clause);

  mlir::LogicalResult
  lowerImportClause(const ast::modelica::ImportClause &clause);

  mlir::LogicalResult
  lowerExternalClause(const ast::modelica::ExternalClause &clause);

  /// Lower the array subscripts of a declaration into dimension attributes,
  /// computing the expressions within the given region.
  mlir::LogicalResult lowerDimensions(
      llvm::ArrayRef<const ast::modelica::Expression *> subscripts,
      llvm::SmallVectorImpl<mlir::Attribute> &dimensions,
      mlir::Region &region, mlir::Location location);

  /// Lower the arguments of a class modification into the given region.
  mlir::LogicalResult
  lowerClassModification(const ast::modelica::ClassModification &modification,
                         mlir::Region &region);

  /// Lower a modification into the given region.
  mlir::LogicalResult
  lowerModification(const ast::modelica::Modification &modification,
                    mlir::Region &region);

  mlir::LogicalResult lowerArgument(const ast::modelica::Argument &argument);

  //===-------------------------------------------------------------------===//
  // Equations and statements (Equations.cpp)
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult
  lowerEquationSection(const ast::modelica::EquationSection &section);

  mlir::LogicalResult
  lowerAlgorithmSection(const ast::modelica::AlgorithmSection &section);

  mlir::LogicalResult lowerEquation(const ast::modelica::Equation &equation);

  mlir::LogicalResult
  lowerEquations(const ast::modelica::Children<ast::modelica::Equation> &equations);

  mlir::LogicalResult
  lowerStatement(const ast::modelica::Statement &statement);

  mlir::LogicalResult lowerStatements(
      const ast::modelica::Children<ast::modelica::Statement> &statements);

  //===-------------------------------------------------------------------===//
  // Expressions (Expressions.cpp)
  //===-------------------------------------------------------------------===//

  /// Lower an expression producing exactly one value.
  mlir::FailureOr<mlir::Value>
  lowerExpression(const ast::modelica::Expression &expression);

  /// Lower an expression possibly producing multiple values, as function
  /// calls with multiple outputs.
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  lowerExpressionResults(const ast::modelica::Expression &expression);

  mlir::FailureOr<mlir::Value>
  lowerComponentReference(const ast::modelica::ComponentReference &reference);

  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  lowerCall(const ast::modelica::Call &call);

  /// Lower a call to a built-in function or operator, if the callee is one.
  /// Returns std::nullopt if the callee is not a built-in function.
  std::optional<mlir::FailureOr<llvm::SmallVector<mlir::Value>>>
  lowerBuiltinCall(const ast::modelica::Call &call);

  /// Collect the arguments of a call to a built-in function, placing the
  /// named arguments according to the given parameter names. Trailing
  /// arguments may be omitted, as long as at least the given number of
  /// arguments is present.
  mlir::FailureOr<llvm::SmallVector<const ast::modelica::Expression *>>
  getBuiltinArguments(const ast::modelica::Call &call,
                      llvm::ArrayRef<llvm::StringRef> parameters,
                      size_t minArguments);

  /// Lower the subscripts applied to a value.
  mlir::FailureOr<mlir::Value> lowerSubscripts(
      mlir::Value array,
      const ast::modelica::Children<ast::modelica::Expression> &subscripts,
      mlir::Location location);

  /// Convert an index, or a vector of indices, from the one-based indexing
  /// of the language to the zero-based one of the dialect. Boolean and
  /// enumeration indices are left unchanged.
  mlir::FailureOr<mlir::Value> toZeroBasedIndex(mlir::Value index,
                                                mlir::Location location);

  /// Lower the access to a member of an object.
  mlir::FailureOr<mlir::Value> lowerMemberAccess(mlir::Value object,
                                                 llvm::StringRef member,
                                                 mlir::Location location);

  /// Lower an expression into a region terminated by a yield, used for
  /// conditions and ranges.
  mlir::LogicalResult
  lowerIntoRegion(mlir::Region &region,
                  llvm::ArrayRef<const ast::modelica::Expression *> expressions,
                  mlir::Location location);

  /// Create an operation whose result type is inferred, reporting an error
  /// if the inference fails.
  template <typename Op>
  mlir::FailureOr<mlir::Value>
  createInferred(mlir::Location location, mlir::ValueRange operands,
                 llvm::ArrayRef<mlir::NamedAttribute> attributes = {}) {
    llvm::SmallVector<mlir::Type, 1> types;

    if (mlir::failed(Op::inferReturnTypes(
            builder.getContext(), location, operands,
            builder.getDictionaryAttr(attributes), mlir::OpaqueProperties(nullptr),
            mlir::RegionRange(), types))) {
      return mlir::failure();
    }

    return builder.create<Op>(location, types, operands, attributes)
        .getResult();
  }

  /// Iterators visible in the current scope.
  void pushIterator(llvm::StringRef name, mlir::Value value);
  void popIterator();
  mlir::Value lookupIterator(llvm::StringRef name) const;

public:
  mlir::OpBuilder builder;

  /// Whether any error has been reported.
  bool hadErrors = false;
  ClassTree classTree;

  /// Class whose scope is used to resolve the names of the expressions being
  /// lowered.
  ClassNode *scope = nullptr;

  /// Arrays being subscripted, used to lower 'end'.
  llvm::SmallVector<std::pair<mlir::Value, unsigned>> subscriptContext;

private:
  llvm::SmallVector<std::pair<std::string, mlir::Value>> iterators;

  llvm::DenseMap<ClassNode *, mlir::Type> classTypes;
  llvm::DenseMap<mlir::Type, ClassNode *> objectClasses;
  llvm::DenseSet<ClassNode *> classTypesInProgress;
  llvm::DenseMap<std::pair<const ast::modelica::ComponentDeclaration *,
                           ClassNode *>,
                 mlir::Type>
      componentTypes;
};
} // namespace marco::codegen::lowering::modelica

#endif // MARCO_CODEGEN_LOWERING_MODELICA_LOWERERIMPL_H
