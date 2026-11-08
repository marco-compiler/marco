#include "marco/Codegen/Lowering/Modelica/LowererImpl.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::codegen::lowering::modelica;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering::modelica {
//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

namespace {
/// Tag used to pass operation types to generic lambdas.
template <typename T>
struct TypeTag {
  using type = T;
};

mlir::FailureOr<mlir::Value> lowerArrayFor(LowererImpl &lowerer,
                                           const Call &call);

/// Follow the short class definitions aliasing other classes.
ClassNode *followAliases(ClassTree &classTree, ClassNode *classNode) {
  llvm::SmallPtrSet<ClassNode *, 4> visited;

  while (classNode && classNode->getDefinition() &&
         classNode->getDefinition()->getForm() ==
             ClassDefinition::Form::Short &&
         visited.insert(classNode).second) {
    classNode = classTree.getShortClassBase(classNode);
  }

  return classNode;
}

/// Get the type resulting from the combination of the values of the branches
/// of an if-expression, applying the implicit conversion of Integer values
/// to Real ones.
mlir::Type combineTypes(mlir::Type lhs, mlir::Type rhs) {
  if (isCoercibleTo(lhs, rhs)) {
    return rhs;
  }

  if (isCoercibleTo(rhs, lhs)) {
    return lhs;
  }

  // Arrays with different sizes are allowed if the condition is evaluable,
  // in which case the size is known only after its evaluation.
  llvm::ArrayRef<int64_t> lhsShape = getShape(lhs);
  llvm::ArrayRef<int64_t> rhsShape = getShape(rhs);

  if (lhsShape.empty() || lhsShape.size() != rhsShape.size()) {
    return nullptr;
  }

  llvm::SmallVector<int64_t> shape;

  for (auto [lhsDim, rhsDim] : llvm::zip(lhsShape, rhsShape)) {
    shape.push_back(lhsDim == rhsDim ? lhsDim : mlir::ShapedType::kDynamic);
  }

  mlir::Type lhsType = getValueType(shape, mlir::getElementTypeOrSelf(lhs));
  mlir::Type rhsType = getValueType(shape, mlir::getElementTypeOrSelf(rhs));

  if (isCoercibleTo(lhsType, rhsType)) {
    return rhsType;
  }

  if (isCoercibleTo(rhsType, lhsType)) {
    return lhsType;
  }

  return nullptr;
}

/// Get whether a class is an external object.
bool isExternalObject(ClassTree &classTree, ClassNode *classNode) {
  llvm::SmallPtrSet<ClassNode *, 8> visited;
  llvm::SmallVector<ClassNode *> worklist = {classNode};

  while (!worklist.empty()) {
    ClassNode *current = worklist.pop_back_val();

    if (!current || !visited.insert(current).second) {
      continue;
    }

    if (current->getPredefined() == PredefinedClass::ExternalObject) {
      return true;
    }

    llvm::append_range(worklist, classTree.getBaseClasses(current));
  }

  return false;
}
} // namespace

mlir::FailureOr<llvm::SmallVector<const Expression *>>
LowererImpl::getBuiltinArguments(const Call &call,
                                 llvm::ArrayRef<llvm::StringRef> parameters,
                                 size_t minArguments) {
  mlir::Location location = loc(call);
  llvm::SmallVector<const Expression *> arguments(parameters.size(), nullptr);

  if (call.getArguments().size() > parameters.size()) {
    return emitError(location, "too many arguments");
  }

  for (auto [index, argument] : llvm::enumerate(call.getArguments())) {
    arguments[index] = argument.get();
  }

  for (const Child<NamedArgument> &argument : call.getNamedArguments()) {
    auto it = llvm::find(parameters, argument->getName());

    if (it == parameters.end()) {
      return emitError(location,
                       "unknown argument '" + argument->getName() + "'");
    }

    arguments[std::distance(parameters.begin(), it)] = argument->getValue();
  }

  // Drop the trailing omitted arguments.
  while (!arguments.empty() && !arguments.back()) {
    arguments.pop_back();
  }

  // Optional arguments may be omitted, while the mandatory ones are
  // required. Omitted arguments are null.
  if (arguments.size() < minArguments ||
      llvm::is_contained(llvm::ArrayRef(arguments).take_front(minArguments),
                         nullptr)) {
    return emitError(location, "missing arguments");
  }

  return arguments;
}

mlir::LogicalResult
LowererImpl::lowerIntoRegion(mlir::Region &region,
                             llvm::ArrayRef<const Expression *> expressions,
                             mlir::Location location) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (region.empty()) {
    region.emplaceBlock();
  }

  builder.setInsertionPointToEnd(&region.front());
  llvm::SmallVector<mlir::Value> values;

  for (const Expression *expression : expressions) {
    auto value = lowerExpression(*expression);

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    values.push_back(*value);
  }

  builder.create<YieldOp>(location, values);
  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Expressions
//===---------------------------------------------------------------------===//

mlir::FailureOr<mlir::Value>
LowererImpl::lowerExpression(const Expression &expression) {
  auto results = lowerExpressionResults(expression);

  if (mlir::failed(results)) {
    return mlir::failure();
  }

  if (results->empty()) {
    return emitError(loc(expression), "the expression has no value");
  }

  // The first output of functions with multiple outputs is used.
  return (*results)[0];
}

namespace {
mlir::FailureOr<mlir::Value> createConstant(mlir::OpBuilder &builder,
                                            mlir::Location location,
                                            mlir::TypedAttr value) {
  return builder.create<ConstantOp>(location, value).getResult();
}
} // namespace

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
LowererImpl::lowerExpressionResults(const Expression &expression) {
  mlir::Location location = loc(expression);
  mlir::MLIRContext *context = builder.getContext();

  auto single = [](mlir::FailureOr<mlir::Value> value)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value>> {
    if (mlir::failed(value)) {
      return mlir::failure();
    }

    return llvm::SmallVector<mlir::Value>{*value};
  };

  auto unitAttr = [&](const std::optional<std::string> &unit) {
    return unit ? builder.getStringAttr(*unit) : mlir::StringAttr();
  };

  if (auto literal = expression.dyn_cast<IntegerLiteral>()) {
    return single(createConstant(
        builder, location,
        mlir::modelica::IntegerAttr::get(context, IntegerType::get(context),
                                         literal->getValue(),
                                         unitAttr(literal->getUnit()))));
  }

  if (auto literal = expression.dyn_cast<RealLiteral>()) {
    return single(createConstant(
        builder, location,
        RealAttr::get(context, RealType::get(context),
                      llvm::APFloat(literal->getValue()),
                      unitAttr(literal->getUnit()))));
  }

  if (auto literal = expression.dyn_cast<StringLiteral>()) {
    return single(createConstant(
        builder, location, StringValueAttr::get(context, literal->getValue())));
  }

  if (auto literal = expression.dyn_cast<BooleanLiteral>()) {
    return single(createConstant(
        builder, location, BooleanAttr::get(context, literal->getValue())));
  }

  if (expression.isa<Time>()) {
    return single(createInferred<TimeOp>(location, {}));
  }

  if (expression.isa<End>()) {
    if (subscriptContext.empty()) {
      return emitError(location, "'end' used outside of array subscripts");
    }

    auto [array, dimension] = subscriptContext.back();

    mlir::Value index = builder.create<ConstantOp>(
        location, mlir::modelica::IntegerAttr::get(
                      context, static_cast<int64_t>(dimension)));

    return single(createInferred<SizeOp>(location, {array, index}));
  }

  if (expression.isa<Colon>()) {
    return emitError(location, "':' can be used only as array subscript");
  }

  if (auto reference = expression.dyn_cast<ComponentReference>()) {
    return single(lowerComponentReference(*reference));
  }

  if (auto operation = expression.dyn_cast<UnaryOperation>()) {
    auto operand = lowerExpression(*operation->getOperand());

    if (mlir::failed(operand)) {
      return mlir::failure();
    }

    switch (operation->getOperator()) {
    case UnaryOperator::Plus:
    case UnaryOperator::PlusEW:
      return single(operand);
    case UnaryOperator::Minus:
    case UnaryOperator::MinusEW:
      return single(createInferred<NegateOp>(location, *operand));
    case UnaryOperator::Not:
      return single(createInferred<NotOp>(location, *operand));
    }
  }

  if (auto operation = expression.dyn_cast<BinaryOperation>()) {
    auto lhs = lowerExpression(*operation->getLhs());

    if (mlir::failed(lhs)) {
      return mlir::failure();
    }

    auto rhs = lowerExpression(*operation->getRhs());

    if (mlir::failed(rhs)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value, 2> operands = {*lhs, *rhs};

    switch (operation->getOperator()) {
    case BinaryOperator::Add:
      return single(createInferred<AddOp>(location, operands));
    case BinaryOperator::AddEW:
      return single(createInferred<AddEWOp>(location, operands));
    case BinaryOperator::Sub:
      return single(createInferred<SubOp>(location, operands));
    case BinaryOperator::SubEW:
      return single(createInferred<SubEWOp>(location, operands));
    case BinaryOperator::Mul:
      return single(createInferred<MulOp>(location, operands));
    case BinaryOperator::MulEW:
      return single(createInferred<MulEWOp>(location, operands));
    case BinaryOperator::Div:
      return single(createInferred<DivOp>(location, operands));
    case BinaryOperator::DivEW:
      return single(createInferred<DivEWOp>(location, operands));
    case BinaryOperator::Pow:
      return single(createInferred<PowOp>(location, operands));
    case BinaryOperator::PowEW:
      return single(createInferred<PowEWOp>(location, operands));
    case BinaryOperator::Equal:
      return single(createInferred<EqOp>(location, operands));
    case BinaryOperator::NotEqual:
      return single(createInferred<NotEqOp>(location, operands));
    case BinaryOperator::Less:
      return single(createInferred<LtOp>(location, operands));
    case BinaryOperator::LessEqual:
      return single(createInferred<LteOp>(location, operands));
    case BinaryOperator::Greater:
      return single(createInferred<GtOp>(location, operands));
    case BinaryOperator::GreaterEqual:
      return single(createInferred<GteOp>(location, operands));
    case BinaryOperator::And:
      return single(createInferred<AndOp>(location, operands));
    case BinaryOperator::Or:
      return single(createInferred<OrOp>(location, operands));
    }
  }

  if (auto range = expression.dyn_cast<Range>()) {
    llvm::SmallVector<mlir::Value, 3> operands;

    for (const Expression *operand :
         {range->getStart(), range->getStop(), range->getStep()}) {
      if (!operand) {
        continue;
      }

      auto value = lowerExpression(*operand);

      if (mlir::failed(value)) {
        return mlir::failure();
      }

      operands.push_back(*value);
    }

    return single(createInferred<RangeOp>(location, operands));
  }

  if (auto ifExpression = expression.dyn_cast<IfExpression>()) {
    // Lower the branches from the last one, nesting each of them into the
    // else region of the previous one.
    std::function<mlir::FailureOr<mlir::Value>(size_t)> lowerBranch =
        [&](size_t index) -> mlir::FailureOr<mlir::Value> {
      auto condition = lowerExpression(*ifExpression->getConditions()[index]);

      if (mlir::failed(condition)) {
        return mlir::failure();
      }

      // The result type is set once the branches are lowered.
      auto op = builder.create<IfExprOp>(location, builder.getNoneType(),
                                         *condition);

      auto lowerRegion = [&](mlir::Region &region,
                             auto lowerValue) -> mlir::FailureOr<mlir::Type> {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&region.emplaceBlock());
        mlir::FailureOr<mlir::Value> value = lowerValue();

        if (mlir::failed(value)) {
          return mlir::failure();
        }

        builder.create<YieldOp>(location, *value);
        return value->getType();
      };

      auto thenType = lowerRegion(op.getThenRegion(), [&]() {
        return lowerExpression(*ifExpression->getValues()[index]);
      });

      auto elseType = lowerRegion(op.getElseRegion(), [&]() {
        if (index + 1 < ifExpression->getConditions().size()) {
          return lowerBranch(index + 1);
        }

        return lowerExpression(*ifExpression->getElseValue());
      });

      if (mlir::failed(thenType) || mlir::failed(elseType)) {
        op.erase();
        return mlir::failure();
      }

      mlir::Type resultType = combineTypes(*thenType, *elseType);

      if (!resultType) {
        op.erase();
        return emitError(location, "the branches of the if-expression have "
                                   "incompatible types");
      }

      op.getResult().setType(resultType);
      return op.getResult();
    };

    return single(lowerBranch(0));
  }

  if (auto call = expression.dyn_cast<Call>()) {
    return lowerCall(*call);
  }

  if (auto constructor = expression.dyn_cast<ArrayConstructor>()) {
    if (!constructor->getIterators().empty()) {
      Call call(constructor->getLocation());
      auto callee = std::make_unique<ComponentReference>(constructor->getLocation());
      callee->addPart({"array", {}, constructor->getLocation()});
      call.setCallee(std::move(callee));

      call.addArgument(std::unique_ptr<Expression>(static_cast<Expression *>(
          constructor->getElements()[0]->clone().release())));

      for (const Child<ForIndex> &iterator : constructor->getIterators()) {
        call.addIterator(std::unique_ptr<ForIndex>(
            static_cast<ForIndex *>(iterator->clone().release())));
      }

      // The array constructor with iterators is the same as the array
      // function with iterators.
      call.setParentOfChildren();
      return single(lowerArrayFor(*this, call));
    }

    llvm::SmallVector<mlir::Value> elements;

    for (const Child<Expression> &element : constructor->getElements()) {
      auto value = lowerExpression(*element);

      if (mlir::failed(value)) {
        return mlir::failure();
      }

      elements.push_back(*value);
    }

    return single(createInferred<ArrayOp>(location, elements));
  }

  if (auto concatenation = expression.dyn_cast<ArrayConcatenation>()) {
    // [A, B; C, D] is cat(1, cat(2, A, B), cat(2, C, D)), with the arrays
    // promoted to at least two dimensions.
    auto promote = [&](mlir::Value value,
                       int64_t rank) -> mlir::FailureOr<mlir::Value> {
      llvm::SmallVector<int64_t> shape(getShape(value.getType()));

      if (static_cast<int64_t>(shape.size()) >= rank) {
        return value;
      }

      shape.resize(rank, 1);

      mlir::Type type =
          getValueType(shape, mlir::getElementTypeOrSelf(value.getType()));

      return builder
          .create<PromoteOp>(location, type, value,
                             builder.getI64IntegerAttr(rank))
          .getResult();
    };

    auto concatenate = [&](llvm::ArrayRef<mlir::Value> values,
                           int64_t dimension) -> mlir::FailureOr<mlir::Value> {
      int64_t rank = 2;

      for (mlir::Value value : values) {
        rank = std::max<int64_t>(rank, getShape(value.getType()).size());
      }

      // The dimension is zero-based in the dialect.
      llvm::SmallVector<mlir::Value> operands;
      operands.push_back(builder.create<ConstantOp>(
          location, mlir::modelica::IntegerAttr::get(context, dimension - 1)));

      for (mlir::Value value : values) {
        auto promoted = promote(value, rank);

        if (mlir::failed(promoted)) {
          return mlir::failure();
        }

        operands.push_back(*promoted);
      }

      if (operands.size() == 2) {
        return operands[1];
      }

      return createInferred<CatOp>(location, operands);
    };

    llvm::SmallVector<mlir::Value> rows;

    for (const Children<Expression> &row : concatenation->getRows()) {
      llvm::SmallVector<mlir::Value> elements;

      for (const Child<Expression> &element : row) {
        auto value = lowerExpression(*element);

        if (mlir::failed(value)) {
          return mlir::failure();
        }

        elements.push_back(*value);
      }

      auto rowValue = concatenate(elements, 2);

      if (mlir::failed(rowValue)) {
        return mlir::failure();
      }

      rows.push_back(*rowValue);
    }

    return single(concatenate(rows, 1));
  }

  if (expression.isa<OutputList>()) {
    return emitError(location, "lists of outputs can be used only as "
                               "left-hand side of equations and assignments");
  }

  if (auto subscripted = expression.dyn_cast<Subscripted>()) {
    auto base = lowerExpression(*subscripted->getBase());

    if (mlir::failed(base)) {
      return mlir::failure();
    }

    return single(
        lowerSubscripts(*base, subscripted->getSubscripts(), location));
  }

  if (auto access = expression.dyn_cast<MemberAccess>()) {
    auto base = lowerExpression(*access->getBase());

    if (mlir::failed(base)) {
      return mlir::failure();
    }

    return single(lowerMemberAccess(*base, access->getMember(), location));
  }

  if (expression.isa<FunctionPartialApplication>()) {
    return emitError(location,
                     "function partial application is not supported yet");
  }

  return emitError(location, "unexpected expression");
}

//===---------------------------------------------------------------------===//
// Component references
//===---------------------------------------------------------------------===//

namespace {
/// Get the values of a Boolean or enumeration type, used as range of
/// iterators.
mlir::FailureOr<mlir::Value> lowerTypeAsRange(LowererImpl &lowerer,
                                              ClassNode *classNode,
                                              mlir::Location location) {
  mlir::OpBuilder &builder = lowerer.builder;
  mlir::MLIRContext *context = builder.getContext();
  ClassNode *base = followAliases(lowerer.classTree, classNode);

  if (base && base->getPredefined() == PredefinedClass::Boolean) {
    auto type = mlir::RankedTensorType::get({2}, BooleanType::get(context));

    auto value = ArrayValueAttr::get(
        context, type,
        {BooleanAttr::get(context, false), BooleanAttr::get(context, true)});

    return builder.create<ConstantOp>(location, value).getResult();
  }

  if (base && base->isEnumeration() &&
      !base->getEnumerationLiterals().empty()) {
    auto enumType = lowerer.getClassType(base, location);

    if (mlir::failed(enumType)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Attribute> literals;

    for (auto [index, literal] :
         llvm::enumerate(base->getEnumerationLiterals())) {
      literals.push_back(EnumLiteralAttr::get(context, *enumType, literal,
                                              static_cast<int64_t>(index) + 1));
    }

    auto type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(literals.size())}, *enumType);

    return builder
        .create<ConstantOp>(location, ArrayValueAttr::get(context, type, literals))
        .getResult();
  }

  return lowerer.emitError(location, "'" + classNode->getPathString() +
                                         "' is a class, not a value");
}
} // namespace

mlir::FailureOr<mlir::Value>
LowererImpl::lowerComponentReference(const ComponentReference &reference) {
  mlir::Location location = loc(reference);
  llvm::ArrayRef<ComponentReference::Part> parts = reference.getParts();
  mlir::Value value;
  size_t index = 0;

  auto applySubscripts = [&](const ComponentReference::Part &part)
      -> mlir::LogicalResult {
    if (part.subscripts.empty()) {
      return mlir::success();
    }

    auto subscripted = lowerSubscripts(value, part.subscripts, loc(part.location));

    if (mlir::failed(subscripted)) {
      return mlir::failure();
    }

    value = *subscripted;
    return mlir::success();
  };

  if (!reference.isGlobal()) {
    value = lookupIterator(parts[0].name);
  }

  if (value) {
    if (mlir::failed(applySubscripts(parts[0]))) {
      return mlir::failure();
    }

    index = 1;
  } else {
    std::optional<Element> element =
        reference.isGlobal()
            ? classTree.lookupMember(classTree.getRoot(), parts[0].name)
            : classTree.lookup(scope, parts[0].name);

    if (!element) {
      return emitError(location, "unknown name '" + parts[0].name + "'");
    }

    if (element->isComponent()) {
      auto type = getComponentType(*element);

      if (mlir::failed(type)) {
        return mlir::failure();
      }

      value = builder.create<RefOp>(location, *type, parts[0].name);

      if (mlir::failed(applySubscripts(parts[0]))) {
        return mlir::failure();
      }

      index = 1;
    } else {
      // The name starts with a class: look up the
      // following parts among the elements of the classes, until a component
      // or an enumeration literal is found.
      ClassNode *classNode = element->classNode;
      index = 1;

      while (!value) {
        if (!parts[index - 1].subscripts.empty()) {
          return emitError(location, "classes can't be subscripted");
        }

        if (index == parts.size()) {
          return lowerTypeAsRange(*this, classNode, location);
        }

        llvm::StringRef name = parts[index].name;
        ClassNode *enumeration = followAliases(classTree, classNode);

        if (enumeration && enumeration->isEnumeration()) {
          auto literals = enumeration->getEnumerationLiterals();
          auto it = llvm::find(literals, name);

          if (it == literals.end()) {
            return emitError(location, "unknown literal '" + name +
                                           "' of enumeration '" +
                                           enumeration->getPathString() + "'");
          }

          auto type = getClassType(enumeration, location);

          if (mlir::failed(type)) {
            return mlir::failure();
          }

          value = builder.create<ConstantOp>(
              location,
              EnumLiteralAttr::get(builder.getContext(), *type, name,
                                   std::distance(literals.begin(), it) + 1));

          ++index;
          break;
        }

        auto member = classTree.lookupMember(classNode, name);

        if (!member) {
          return emitError(location, "unknown element '" + name +
                                         "' of class '" +
                                         classNode->getPathString() + "'");
        }

        if (member->isClass()) {
          classNode = member->classNode;
          ++index;
          continue;
        }

        auto type = getComponentType(*member);

        if (mlir::failed(type)) {
          return mlir::failure();
        }

        Name path;
        path.global = reference.isGlobal();

        for (size_t i = 0; i <= index; ++i) {
          path.parts.push_back(parts[i].name);
        }

        value = builder.create<RefOp>(location, *type, getSymbolRef(path));

        if (mlir::failed(applySubscripts(parts[index]))) {
          return mlir::failure();
        }

        ++index;
      }
    }
  }

  for (; index < parts.size(); ++index) {
    auto member = lowerMemberAccess(value, parts[index].name, location);

    if (mlir::failed(member)) {
      return mlir::failure();
    }

    value = *member;

    if (mlir::failed(applySubscripts(parts[index]))) {
      return mlir::failure();
    }
  }

  return value;
}

mlir::FailureOr<mlir::Value>
LowererImpl::lowerMemberAccess(mlir::Value object, llvm::StringRef member,
                               mlir::Location location) {
  ClassNode *classNode = getObjectClass(object.getType());

  if (!classNode) {
    return emitError(location, "member '" + member +
                                   "' accessed on a value that is not an "
                                   "object");
  }

  auto element = classTree.lookupMember(classNode, member);

  if (!element || !element->isComponent()) {
    return emitError(location, "unknown component '" + member +
                                   "' of class '" +
                                   classNode->getPathString() + "'");
  }

  auto memberType = getComponentType(*element);

  if (mlir::failed(memberType)) {
    return mlir::failure();
  }

  // Slice operation on arrays of objects.
  llvm::SmallVector<int64_t> shape(getShape(object.getType()));
  llvm::append_range(shape, getShape(*memberType));

  mlir::Type type =
      getValueType(shape, mlir::getElementTypeOrSelf(*memberType));

  return builder
      .create<MemberOp>(location, type, object,
                        mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                     member))
      .getResult();
}

mlir::FailureOr<mlir::Value>
LowererImpl::lowerSubscripts(mlir::Value array,
                             const Children<Expression> &subscripts,
                             mlir::Location location) {
  auto arrayType = mlir::dyn_cast<mlir::RankedTensorType>(array.getType());

  if (!arrayType) {
    return emitError(location, "subscripts applied to a scalar value");
  }

  if (static_cast<int64_t>(subscripts.size()) > arrayType.getRank()) {
    return emitError(location, "too many subscripts for an array of rank " +
                                   llvm::Twine(arrayType.getRank()));
  }

  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<bool> colons;
  llvm::SmallVector<int64_t> shape;

  for (auto [dimension, subscript] : llvm::enumerate(subscripts)) {
    if (subscript->isa<Colon>()) {
      colons.push_back(true);
      shape.push_back(arrayType.getDimSize(dimension));
      continue;
    }

    subscriptContext.emplace_back(array, dimension);
    auto index = lowerExpression(*subscript);
    subscriptContext.pop_back();

    if (mlir::failed(index)) {
      return mlir::failure();
    }

    index = toZeroBasedIndex(*index, location);

    if (mlir::failed(index)) {
      return mlir::failure();
    }

    colons.push_back(false);
    indices.push_back(*index);

    // Vector indices select multiple elements.
    if (auto indexType = mlir::dyn_cast<mlir::RankedTensorType>(index->getType())) {
      if (indexType.getRank() != 1) {
        return emitError(location, "array indices must be vectors");
      }

      shape.push_back(indexType.getDimSize(0));
    }
  }

  for (int64_t dimension = subscripts.size(); dimension < arrayType.getRank();
       ++dimension) {
    shape.push_back(arrayType.getDimSize(dimension));
  }

  mlir::Type type = getValueType(shape, arrayType.getElementType());

  return builder
      .create<SubscriptOp>(location, type, array, indices,
                           builder.getDenseBoolArrayAttr(colons))
      .getResult();
}

mlir::FailureOr<mlir::Value>
LowererImpl::toZeroBasedIndex(mlir::Value index, mlir::Location location) {
  if (!mlir::isa<IntegerType>(mlir::getElementTypeOrSelf(index.getType()))) {
    return index;
  }

  mlir::MLIRContext *context = builder.getContext();

  auto eraseIfUnused = [](mlir::Value value) {
    if (mlir::Operation *op = value.getDefiningOp(); op && op->use_empty()) {
      op->erase();
    }
  };

  // Literal indices are shifted directly.
  if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
    if (auto integerAttr =
            mlir::dyn_cast<mlir::modelica::IntegerAttr>(constantOp.getValue());
        integerAttr && !integerAttr.getUnit()) {
      mlir::Value result =
          builder
              .create<ConstantOp>(location,
                                  mlir::modelica::IntegerAttr::get(
                                      context, integerAttr.getValue() - 1))
              .getResult();

      eraseIfUnused(index);
      return result;
    }
  }

  // Ranges are kept as such, so that they can be recognized as slices.
  if (auto rangeOp = index.getDefiningOp<RangeOp>()) {
    mlir::Value start = rangeOp.getStart();
    mlir::Value stop = rangeOp.getStop();
    auto newStart = toZeroBasedIndex(start, location);
    auto newStop = toZeroBasedIndex(stop, location);

    if (mlir::failed(newStart) || mlir::failed(newStop)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value, 3> operands{*newStart, *newStop};

    if (mlir::Value step = rangeOp.getStep()) {
      operands.push_back(step);
    }

    auto result = createInferred<RangeOp>(location, operands);

    if (mlir::failed(result)) {
      return mlir::failure();
    }

    eraseIfUnused(index);
    eraseIfUnused(start);
    eraseIfUnused(stop);
    return result;
  }

  // The elements of vector constructors are shifted one by one, so that
  // literal elements remain constants.
  if (auto arrayOp = index.getDefiningOp<ArrayOp>();
      arrayOp && llvm::none_of(arrayOp.getValues(), [](mlir::Value element) {
        return mlir::isa<mlir::RankedTensorType>(element.getType());
      })) {
    llvm::SmallVector<mlir::Value> elements(arrayOp.getValues());
    llvm::SmallVector<mlir::Value> newElements;

    for (mlir::Value element : elements) {
      auto newElement = toZeroBasedIndex(element, location);

      if (mlir::failed(newElement)) {
        return mlir::failure();
      }

      newElements.push_back(*newElement);
    }

    auto result = createInferred<ArrayOp>(location, newElements);

    if (mlir::failed(result)) {
      return mlir::failure();
    }

    eraseIfUnused(index);

    for (mlir::Value element : elements) {
      eraseIfUnused(element);
    }

    return result;
  }

  mlir::Value one =
      builder
          .create<ConstantOp>(location,
                              mlir::modelica::IntegerAttr::get(context, 1))
          .getResult();

  if (mlir::isa<mlir::RankedTensorType>(index.getType())) {
    return createInferred<SubEWOp>(location, {index, one});
  }

  return createInferred<SubOp>(location, {index, one});
}

//===---------------------------------------------------------------------===//
// Function calls
//===---------------------------------------------------------------------===//

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
LowererImpl::lowerCall(const Call &call) {
  if (auto builtin = lowerBuiltinCall(call)) {
    return *builtin;
  }

  mlir::Location location = loc(call);
  Name name = call.getCallee()->getName();

  for (const ComponentReference::Part &part : call.getCallee()->getParts()) {
    if (!part.subscripts.empty()) {
      return emitError(location, "subscripted function names are not "
                                 "supported");
    }
  }

  auto element = classTree.lookup(scope, name);

  if (!element) {
    return emitError(location, "unknown function '" + name.toString() + "'");
  }

  if (!element->isClass()) {
    return emitError(location, "'" + name.toString() + "' is not a function");
  }

  ClassNode *classNode = element->classNode;
  ClassNode *base = followAliases(classTree, classNode);

  if (!call.getIterators().empty()) {
    return emitError(location, "iterators can be used only with reductions "
                               "and array constructors");
  }

  // Conversion from Integer to enumeration.
  if (base && base->isEnumeration()) {
    if (call.getArguments().size() != 1 || !call.getNamedArguments().empty()) {
      return emitError(location, "enumeration conversions require exactly "
                                 "one argument");
    }

    auto argument = lowerExpression(*call.getArguments()[0]);
    auto enumType = getClassType(base, location);

    if (mlir::failed(argument) || mlir::failed(enumType)) {
      return mlir::failure();
    }

    mlir::Type type = getValueType(getShape(argument->getType()), *enumType);

    return llvm::SmallVector<mlir::Value>{
        builder.create<IntegerToEnumOp>(location, type, *argument).getResult()};
  }

  if (!base || !base->getDefinition()) {
    return emitError(location, "'" + name.toString() + "' is not a function");
  }

  llvm::SmallVector<mlir::Type> resultTypes;

  switch (base->getDefinition()->getRestriction()) {
  case ClassRestriction::Function:
  case ClassRestriction::OperatorFunction: {
    auto types = getFunctionResultTypes(classNode, location);

    if (mlir::failed(types)) {
      return mlir::failure();
    }

    resultTypes = std::move(*types);
    break;
  }

  case ClassRestriction::Class:
    if (!isExternalObject(classTree, base)) {
      return emitError(location, "'" + name.toString() + "' is not a function");
    }

    // Constructor of an external object.
    [[fallthrough]];

  case ClassRestriction::Record:
  case ClassRestriction::OperatorRecord: {
    // Record constructor.
    auto type = getClassType(base, location);

    if (mlir::failed(type)) {
      return mlir::failure();
    }

    resultTypes.push_back(*type);
    break;
  }

  default:
    return emitError(location, "'" + name.toString() + "' is not a function");
  }

  llvm::SmallVector<mlir::Value> args;
  llvm::SmallVector<llvm::StringRef> argNames;

  for (const Child<Expression> &argument : call.getArguments()) {
    auto value = lowerExpression(*argument);

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    args.push_back(*value);
  }

  for (const Child<NamedArgument> &argument : call.getNamedArguments()) {
    auto value = lowerExpression(*argument->getValue());

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    args.push_back(*value);
    argNames.push_back(argument->getName());
  }

  // Automatic vectorization: arguments having more
  // dimensions than the corresponding inputs are iterated over, and the
  // results gain their leading dimensions.
  if (base->getDefinition()->getRestriction() == ClassRestriction::Function ||
      base->getDefinition()->getRestriction() ==
          ClassRestriction::OperatorFunction) {
    llvm::SmallVector<Element> components;
    collectComponents(classNode, components);

    llvm::SmallVector<Element> inputs;

    for (const Element &component : components) {
      if (component.clause->getTypePrefix().causality ==
          ast::modelica::Causality::Input) {
        inputs.push_back(component);
      }
    }

    std::optional<llvm::SmallVector<int64_t>> foreachShape;

    for (auto [index, arg] : llvm::enumerate(args)) {
      const Element *input = nullptr;

      if (index < call.getArguments().size()) {
        if (index < inputs.size()) {
          input = &inputs[index];
        }
      } else {
        llvm::StringRef argName = argNames[index - call.getArguments().size()];

        for (const Element &candidate : inputs) {
          if (candidate.declaration->getName() == argName) {
            input = &candidate;
          }
        }
      }

      if (!input) {
        continue;
      }

      auto inputType = getComponentType(*input);

      if (mlir::failed(inputType)) {
        return mlir::failure();
      }

      llvm::ArrayRef<int64_t> argShape = getShape(arg.getType());
      size_t inputRank = getShape(*inputType).size();

      if (argShape.size() <= inputRank) {
        continue;
      }

      llvm::SmallVector<int64_t> shape(
          argShape.take_front(argShape.size() - inputRank));

      if (foreachShape && foreachShape->size() != shape.size()) {
        return emitError(location, "the vectorized arguments have different "
                                   "numbers of dimensions");
      }

      foreachShape = shape;
    }

    if (foreachShape) {
      for (mlir::Type &type : resultTypes) {
        llvm::SmallVector<int64_t> shape(*foreachShape);
        llvm::append_range(shape, getShape(type));
        type = getValueType(shape, mlir::getElementTypeOrSelf(type));
      }
    }
  }

  auto callOp = builder.create<CallOp>(
      location, resultTypes, getSymbolRef(name), args,
      argNames.empty() ? nullptr : builder.getStrArrayAttr(argNames));

  return llvm::SmallVector<mlir::Value>(callOp.getResults());
}

//===---------------------------------------------------------------------===//
// Built-in functions and operators
//===---------------------------------------------------------------------===//

namespace {
/// Lower an iterator-based expression (array constructor or reduction),
/// creating the operation through the given callback once the ranges are
/// lowered.
mlir::FailureOr<mlir::Value> lowerIteratorExpression(
    LowererImpl &lowerer, const Call &call,
    llvm::function_ref<mlir::Operation *(llvm::ArrayRef<mlir::Value> ranges)>
        createOp,
    llvm::function_ref<mlir::Type(llvm::ArrayRef<mlir::Value> ranges,
                                  mlir::Type yieldedType)>
        getResultType) {
  mlir::OpBuilder &builder = lowerer.builder;
  mlir::Location location = lowerer.loc(call);

  if (call.getArguments().size() != 1 || !call.getNamedArguments().empty()) {
    return lowerer.emitError(location,
                             "iterator expressions require exactly one "
                             "expression");
  }

  llvm::SmallVector<mlir::Value> ranges;

  for (const Child<ForIndex> &iterator : call.getIterators()) {
    if (!iterator->getRange()) {
      return lowerer.emitError(lowerer.loc(*iterator),
                               "implicit iteration ranges are not supported "
                               "yet");
    }

    auto range = lowerer.lowerExpression(*iterator->getRange());

    if (mlir::failed(range)) {
      return mlir::failure();
    }

    auto rangeType = mlir::dyn_cast<mlir::RankedTensorType>(range->getType());

    if (!rangeType || rangeType.getRank() != 1) {
      return lowerer.emitError(lowerer.loc(*iterator),
                               "the range of an iterator must be a vector");
    }

    ranges.push_back(*range);
  }

  mlir::Operation *op = createOp(ranges);
  mlir::Block *body = &op->getRegion(0).emplaceBlock();

  for (auto [iterator, range] : llvm::zip(call.getIterators(), ranges)) {
    mlir::Value argument = body->addArgument(
        mlir::cast<mlir::RankedTensorType>(range.getType()).getElementType(),
        lowerer.loc(*iterator));

    lowerer.pushIterator(iterator->getName(), argument);
  }

  mlir::FailureOr<mlir::Value> value;

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    value = lowerer.lowerExpression(*call.getArguments()[0]);

    if (mlir::succeeded(value)) {
      builder.create<YieldOp>(location, *value);
    }
  }

  for (size_t i = 0; i < ranges.size(); ++i) {
    lowerer.popIterator();
  }

  if (mlir::failed(value)) {
    op->erase();
    return mlir::failure();
  }

  op->getResult(0).setType(getResultType(ranges, value->getType()));
  return op->getResult(0);
}

/// Lower an array constructor with iterators. The last iterator varies along
/// the outermost dimension.
mlir::FailureOr<mlir::Value> lowerArrayFor(LowererImpl &lowerer,
                                           const Call &call) {
  mlir::Location location = lowerer.loc(call);

  return lowerIteratorExpression(
      lowerer, call,
      [&](llvm::ArrayRef<mlir::Value> ranges) {
        return lowerer.builder
            .create<ArrayForOp>(location, lowerer.builder.getNoneType(), ranges)
            .getOperation();
      },
      [&](llvm::ArrayRef<mlir::Value> ranges, mlir::Type yieldedType) {
        llvm::SmallVector<int64_t> shape;

        for (mlir::Value range : llvm::reverse(ranges)) {
          shape.push_back(
              mlir::cast<mlir::RankedTensorType>(range.getType()).getDimSize(0));
        }

        llvm::append_range(shape, getShape(yieldedType));
        return mlir::RankedTensorType::get(
            shape, mlir::getElementTypeOrSelf(yieldedType));
      });
}

/// Lower a reduction expression.
mlir::FailureOr<mlir::Value> lowerReduction(LowererImpl &lowerer,
                                            const Call &call,
                                            ReductionKind kind) {
  mlir::Location location = lowerer.loc(call);

  return lowerIteratorExpression(
      lowerer, call,
      [&](llvm::ArrayRef<mlir::Value> ranges) {
        return lowerer.builder
            .create<ReductionOp>(location, lowerer.builder.getNoneType(), kind,
                                 ranges)
            .getOperation();
      },
      [](llvm::ArrayRef<mlir::Value>, mlir::Type yieldedType) {
        return yieldedType;
      });
}
} // namespace

std::optional<mlir::FailureOr<llvm::SmallVector<mlir::Value>>>
LowererImpl::lowerBuiltinCall(const Call &call) {
  Name calleeName = call.getCallee()->getName();
  std::string name;

  if (calleeName.global) {
    // Global names may refer to the built-in functions, as in '.sqrt(x)'.
    if (calleeName.parts.size() != 1 ||
        classTree.getRoot()->getChild(calleeName.parts[0])) {
      return std::nullopt;
    }

    name = calleeName.parts[0];
  } else if (calleeName.parts.size() == 1) {
    name = calleeName.parts[0];

    // The operators with function syntax can't be shadowed, while the other
    // functions can be redefined by the user.
    if (name != "der" && name != "initial" && name != "pure") {
      if (auto element = classTree.lookup(scope, name)) {
        if (!element->isClass() ||
            (element->classNode->getPredefined() != PredefinedClass::Integer &&
             element->classNode->getPredefined() != PredefinedClass::String)) {
          return std::nullopt;
        }
      }
    }
  } else if (calleeName.parts.size() == 2) {
    auto element = classTree.lookup(scope, calleeName.parts[0]);

    if (!element || !element->isClass() ||
        element->classNode->getPredefined() != PredefinedClass::Connections) {
      return std::nullopt;
    }

    name = "Connections." + calleeName.parts[1];
  } else {
    return std::nullopt;
  }

  mlir::Location location = loc(call);
  using Result = mlir::FailureOr<llvm::SmallVector<mlir::Value>>;

  auto single = [](mlir::FailureOr<mlir::Value> value) -> Result {
    if (mlir::failed(value)) {
      return mlir::failure();
    }

    return llvm::SmallVector<mlir::Value>{*value};
  };

  // Lower the arguments of the call, given the names of the parameters.
  auto lowerArguments = [&](llvm::ArrayRef<llvm::StringRef> parameters,
                            size_t minArguments)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value>> {
    auto arguments = getBuiltinArguments(call, parameters, minArguments);

    if (mlir::failed(arguments)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> values;

    for (const Expression *argument : *arguments) {
      if (!argument) {
        values.push_back(nullptr);
        continue;
      }

      auto value = lowerExpression(*argument);

      if (mlir::failed(value)) {
        return mlir::failure();
      }

      values.push_back(*value);
    }

    return values;
  };

  // Lower the arguments of variadic functions.
  auto lowerVariadicArguments =
      [&]() -> mlir::FailureOr<llvm::SmallVector<mlir::Value>> {
    if (!call.getNamedArguments().empty()) {
      return emitError(location, "named arguments are not allowed");
    }

    llvm::SmallVector<mlir::Value> values;

    for (const Child<Expression> &argument : call.getArguments()) {
      auto value = lowerExpression(*argument);

      if (mlir::failed(value)) {
        return mlir::failure();
      }

      values.push_back(*value);
    }

    return values;
  };

  // Create an operation whose result type is inferred, with the given
  // parameters.
  // Dimension numbers are one-based in the language and zero-based in the
  // dialect.
  auto toZeroBasedDimension = [&](mlir::Value &dimension) {
    auto result = toZeroBasedIndex(dimension, location);

    if (mlir::failed(result)) {
      return mlir::failure();
    }

    dimension = *result;
    return mlir::success();
  };

  auto inferred = [&](auto tag, llvm::ArrayRef<llvm::StringRef> parameters,
                      size_t minArguments,
                      std::optional<size_t> dimensionOperand =
                          std::nullopt) -> Result {
    using Op = typename decltype(tag)::type;

    if (!call.getIterators().empty()) {
      return emitError(location,
                       "iterators are not allowed for '" + name + "'");
    }

    auto operands = lowerArguments(parameters, minArguments);

    if (mlir::failed(operands)) {
      return mlir::failure();
    }

    if (llvm::is_contained(*operands, nullptr)) {
      return emitError(location, "missing arguments");
    }

    if (dimensionOperand && *dimensionOperand < operands->size() &&
        mlir::failed(toZeroBasedDimension((*operands)[*dimensionOperand]))) {
      return mlir::failure();
    }

    return single(createInferred<Op>(location, *operands));
  };

  auto variadic = [&](auto tag, std::optional<size_t> dimensionOperand =
                                    std::nullopt) -> Result {
    using Op = typename decltype(tag)::type;
    auto operands = lowerVariadicArguments();

    if (mlir::failed(operands)) {
      return mlir::failure();
    }

    if (dimensionOperand && *dimensionOperand < operands->size() &&
        mlir::failed(toZeroBasedDimension((*operands)[*dimensionOperand]))) {
      return mlir::failure();
    }

    return single(createInferred<Op>(location, *operands));
  };

  // Reductions, which can be applied to an array or used with iterators.
  auto reduction = [&](ReductionKind kind, auto binaryTag) -> Result {
    using BinaryOp = typename decltype(binaryTag)::type;

    if (!call.getIterators().empty()) {
      return single(lowerReduction(*this, call, kind));
    }

    auto operands = lowerVariadicArguments();

    if (mlir::failed(operands)) {
      return mlir::failure();
    }

    if (operands->size() == 1) {
      return single(createInferred<ArrayReduceOp>(
          location, *operands,
          {builder.getNamedAttr(
              "kind", ReductionKindAttr::get(builder.getContext(), kind))}));
    }

    if constexpr (!std::is_same_v<BinaryOp, void>) {
      if (operands->size() == 2) {
        return single(createInferred<BinaryOp>(location, *operands));
      }
    }

    return emitError(location, "wrong number of arguments for '" + name + "'");
  };

  auto unary = [&](auto tag) { return inferred(tag, {"x"}, 1); };

  using llvm::StringRef;
  StringRef callee = name;

#define UNARY(NAME, OP)                                                        \
  if (callee == NAME)                                                          \
  return unary(TypeTag<OP>())

  // Numeric functions and conversions.
  UNARY("abs", AbsOp);
  UNARY("sign", SignOp);
  UNARY("sqrt", SqrtOp);

  if (callee == "nthRoot") {
    return inferred(TypeTag<NthRootOp>(), {"x", "n"}, 2);
  }

  if (callee == "Integer") {
    return unary(TypeTag<EnumToIntegerOp>());
  }

  if (callee == "String") {
    auto arguments =
        getBuiltinArguments(call,
                            {"x", "minimumLength", "leftJustified",
                             "significantDigits", "format"},
                            1);

    if (mlir::failed(arguments)) {
      return Result(mlir::failure());
    }

    if (call.getArguments().size() > 1) {
      return emitError(location, "the options of 'String' must be given as "
                                 "named arguments");
    }

    llvm::SmallVector<mlir::Value, 5> values(5, nullptr);

    for (auto [index, argument] : llvm::enumerate(*arguments)) {
      if (argument) {
        auto value = lowerExpression(*argument);

        if (mlir::failed(value)) {
          return Result(mlir::failure());
        }

        values[index] = *value;
      }
    }

    return single(builder
                      .create<ToStringOp>(location,
                                          StringType::get(builder.getContext()),
                                          values[0], values[1], values[2],
                                          values[3], values[4])
                      .getResult());
  }

  // Event-triggering mathematical functions.
  if (callee == "div") {
    return inferred(TypeTag<DivTruncOp>(), {"x", "y"}, 2);
  }

  if (callee == "mod") {
    return inferred(TypeTag<ModOp>(), {"x", "y"}, 2);
  }

  if (callee == "rem") {
    return inferred(TypeTag<RemOp>(), {"x", "y"}, 2);
  }

  UNARY("ceil", CeilOp);
  UNARY("floor", FloorOp);
  UNARY("integer", IntegerOp);

  if (callee == "delay") {
    return inferred(TypeTag<DelayOp>(),
                    {"expr", "delayTime", "delayMax"}, 2);
  }

  if (callee == "spatialDistribution") {
    auto operands = lowerArguments({"in0", "in1", "x", "positiveVelocity",
                                    "initialPoints", "initialValues"},
                                   4);

    if (mlir::failed(operands)) {
      return Result(mlir::failure());
    }

    mlir::Type realType = RealType::get(builder.getContext());
    (*operands).resize(6, nullptr);

    auto op = builder.create<SpatialDistributionOp>(
        location, realType, realType, (*operands)[0], (*operands)[1],
        (*operands)[2], (*operands)[3], (*operands)[4], (*operands)[5]);

    return llvm::SmallVector<mlir::Value>{op.getOut0(), op.getOut1()};
  }

  // Elementary mathematical functions.
  UNARY("sin", SinOp);
  UNARY("cos", CosOp);
  UNARY("tan", TanOp);
  UNARY("asin", AsinOp);
  UNARY("acos", AcosOp);
  UNARY("atan", AtanOp);
  UNARY("sinh", SinhOp);
  UNARY("cosh", CoshOp);
  UNARY("tanh", TanhOp);
  UNARY("exp", ExpOp);
  UNARY("log", LogOp);
  UNARY("log10", Log10Op);

  if (callee == "atan2") {
    return inferred(TypeTag<Atan2Op>(), {"u1", "u2"}, 2);
  }

  // Derivative and special purpose operators.
  UNARY("der", DerOp);
  UNARY("cardinality", CardinalityOp);
  UNARY("inStream", InStreamOp);
  UNARY("actualStream", ActualStreamOp);

  if (callee == "homotopy") {
    return inferred(TypeTag<HomotopyOp>(), {"actual", "simplified"},
                    2);
  }

  if (callee == "semiLinear") {
    return inferred(TypeTag<SemiLinearOp>(),
                    {"x", "k_pos", "k_neg"}, 3);
  }

  if (callee == "getInstanceName") {
    return inferred(TypeTag<GetInstanceNameOp>(), {}, 0);
  }

  // Event-related operators.
  if (callee == "initial") {
    return inferred(TypeTag<InitialOp>(), {}, 0);
  }

  if (callee == "terminal") {
    return inferred(TypeTag<TerminalOp>(), {}, 0);
  }

  UNARY("noEvent", NoEventOp);

  if (callee == "smooth") {
    return inferred(TypeTag<SmoothOp>(), {"p", "expr"}, 2);
  }

  if (callee == "sample") {
    return inferred(TypeTag<SampleOp>(), {"start", "interval"}, 2);
  }

  UNARY("pre", PreOp);
  UNARY("edge", EdgeOp);
  UNARY("change", ChangeOp);

  // The purity marker doesn't affect the value of the expression.
  if (callee == "pure") {
    if (call.getArguments().size() != 1 || !call.getNamedArguments().empty()) {
      return emitError(location, "'pure' requires exactly one argument");
    }

    return lowerExpressionResults(*call.getArguments()[0]);
  }

  // Array dimension and conversion functions.
  UNARY("ndims", NDimsOp);

  if (callee == "size") {
    return inferred(TypeTag<SizeOp>(), {"A", "i"}, 1, 1);
  }

  UNARY("scalar", ScalarOp);
  UNARY("vector", VectorOp);
  UNARY("matrix", MatrixOp);

  // Array constructor functions.
  UNARY("identity", IdentityOp);
  UNARY("diagonal", DiagonalOp);

  if (callee == "zeros") {
    return variadic(TypeTag<ZerosOp>());
  }

  if (callee == "ones") {
    return variadic(TypeTag<OnesOp>());
  }

  if (callee == "fill") {
    return variadic(TypeTag<FillOp>());
  }

  if (callee == "linspace") {
    return inferred(TypeTag<LinspaceOp>(), {"x1", "x2", "n"}, 3);
  }

  if (callee == "cat") {
    return variadic(TypeTag<CatOp>(), 0);
  }

  if (callee == "array") {
    if (!call.getIterators().empty()) {
      return single(lowerArrayFor(*this, call));
    }

    return variadic(TypeTag<ArrayOp>());
  }

  // Reductions.
  if (callee == "min") {
    return reduction(ReductionKind::min, TypeTag<MinOp>());
  }

  if (callee == "max") {
    return reduction(ReductionKind::max, TypeTag<MaxOp>());
  }

  if (callee == "sum") {
    return reduction(ReductionKind::sum, TypeTag<void>());
  }

  if (callee == "product") {
    return reduction(ReductionKind::product, TypeTag<void>());
  }

  // Matrix and vector algebra functions.
  UNARY("transpose", TransposeOp);
  UNARY("symmetric", SymmetricOp);
  UNARY("skew", SkewOp);

  if (callee == "outerProduct") {
    return inferred(TypeTag<OuterProductOp>(), {"x", "y"}, 2);
  }

  if (callee == "cross") {
    return inferred(TypeTag<CrossOp>(), {"x", "y"}, 2);
  }

  // Operators of overconstrained connection graphs.
  if (callee == "Connections.isRoot") {
    return inferred(TypeTag<ConnectionsIsRootOp>(), {"a"}, 1);
  }

  if (callee == "Connections.rooted" || callee == "rooted") {
    return inferred(TypeTag<ConnectionsRootedOp>(), {"a"}, 1);
  }

  if (callee == "Connections.uniqueRootIndices") {
    return inferred(TypeTag<ConnectionsUniqueRootIndicesOp>(),
                    {"roots", "nodes", "message"}, 3);
  }

#undef UNARY

  // Operators usable only as equations or statements.
  if (callee == "assert" || callee == "terminate" || callee == "reinit" ||
      callee.starts_with("Connections.")) {
    return Result(emitError(location, "'" + name +
                                          "' can be used only as equation or "
                                          "statement"));
  }

  return std::nullopt;
}
} // namespace marco::codegen::lowering::modelica
