#include <marco/AST/Passes/RaggedFlatteningPass.h>

#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <variant>

#include <iostream>

using namespace marco;
using namespace marco::ast;

using Shape = RaggedFlattener::Shape;


// helper for visiting the AST with lambdas
template<class... Ts>
struct ASTConstVisitor : Ts... {
  using Ts::operator()...;
  template<typename... Args>
  void visit(const std::unique_ptr<Class>& c, Args... args)
  {
    visit(*c, args...);
  }
  template<typename... Args>
  void visit(const Class& c, Args... args)
  {
    c.visit([&](const auto& obj) {
      return visit(obj, args...);
    });
  }
  template<typename T, typename... Args>
  void visit(const T& t, Args... args)
  {
    (*this)(t, args...);
  }
  template<typename... Args>
  void visit(const Package& p, Args... args)
  {
    (*this)(p, args...);
    for (const auto& c : p.getInnerClasses())
      visit(*c, args...);
  }
  template<typename... Args>
  void visit(const EquationsBlock& b, Args... args)
  {
    (*this)(b, args...);
    for (const auto& e : b.getEquations())
      visit(*e, args...);
    for (const auto& e : b.getForEquations())
      visit(*e, args...);
  }
  template<typename... Args>
  void visit(const Model& m, Args... args)
  {
    (*this)(m, args...);
    for (const auto& member : m.getMembers())
      visit(*member, args...);
    for (const auto& a : m.getAlgorithms())
      visit(*a, args...);
    for (const auto& e : m.getEquationsBlocks())
      visit(*e, args...);
    for (const auto& c : m.getInnerClasses())
      visit(*c, args...);
  }
  template<typename... Args>
  void visit(const Statement& s, Args... args)
  {
    (*this)(s, args...);
    s.visit([&](const auto& obj) {
      visit(obj, args...);
      for (const auto& e : obj)
        visit(*e, args...);
    });
  }
  template<typename... Args>
  void visit(const Equation& e, Args... args)
  {
    (*this)(e, args...);
    visit(*e.getRhsExpression(), args...);
    visit(*e.getLhsExpression(), args...);
  }
  template<typename... Args>
  void visit(const ForEquation& f, Args... args)
  {
    (*this)(f, args...);
    for (const auto& i : f.getInductions())
      visit(*i, args...);
    visit(*f.getEquation(), args...);
  }
  template<typename... Args>
  void visit(const Induction& i, Args... args)
  {
    (*this)(i, args...);
    visit(*i.getBegin(), args...);
    visit(*i.getEnd(), args...);
  }
  template<typename... Args>
  void visit(const Expression& e, Args... args)
  {
    (*this)(e, args...);
    e.visit([&](const auto& obj) {
      visit(obj, args...);
    });
  }
  template<typename... Args>
  void visit(const Operation& o, Args... args)
  {
    (*this)(o, args...);
    for (const auto& a : o)
      visit(*a, args...);
  }
  template<typename... Args>
  void visit(const Call& c, Args... args)
  {
    (*this)(c, args...);
    for (const auto& a : c)
      visit(*a, args...);
  }
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
ASTConstVisitor(Ts...) -> ASTConstVisitor<Ts...>;

long getIntFromExpression(const Expression& e)
{
  auto p = e.dyn_get<Constant>();
  assert(p);
  return p->as<BuiltInType::Integer>();
}


//####// RaggedFlattener //#####//

RaggedFlattener::RaggedFlattener(diagnostic::DiagnosticEngine& diagnostics)
    : Pass(diagnostics)
{
}

Model* RaggedFlattener::getModel()
{
  for (auto it = parentStack.rbegin(); it != parentStack.rend(); it++) {
    if (auto m = static_cast<Model*>(*it))
      return m;
  }
  return nullptr;
}

template<>
bool RaggedFlattener::run<Class>(Class& cls)
{
  return cls.visit([&](auto& obj) {
    using type = decltype(obj);
    using deref = typename std::remove_reference<type>::type;
    using deconst = typename std::remove_const<deref>::type;
    return run<deconst>(cls);
  });
}

bool RaggedFlattener::run(std::unique_ptr<Class>& cls)
{
  //RaggedFlattener starting point
  SymbolTable::ScopeTy scope(symbolTable);
  ShapeTable::ScopeTy shapeScope(shapeTable);

  if (!run<Class>(*cls))
    return false;

  return true;
}

template<>
bool RaggedFlattener::run<PartialDerFunction>(Class& cls)
{
  return true;
}

template<>
bool RaggedFlattener::run<StandardFunction>(Class& cls)
{
  auto* function = cls.get<StandardFunction>();

  // Populate the symbol table
  symbolTable.insert(function->getName(), Symbol(cls));

  return true;
}

static bool isRagged(llvm::ArrayRef<Shape::DimensionSize> dims)
{
  for (const auto& dim : dims)
    if (dim.isRagged())
      return true;

  return false;
}
static bool isRagged(const Type& type)
{

  for (const auto& dim : type) {
    if (dim.hasExpression()) {
      const auto* expr = dim.getExpression();

      if (expr->isa<Array>()) {
        return true;
      }
    }
  }
  return false;
}

static bool isRagged(const Member& member)
{
  return isRagged(member.getType());
}

static Shape::DimensionSize getDimension(const Expression& expr)
{

  if (auto val = expr.dyn_get<Constant>()) {
    return val->as<BuiltInType::Integer>();
  }

  if (auto array = expr.dyn_get<Array>()) {
    llvm::SmallVector<Shape::DimensionSize, 3> dimensions;
    for (const auto& val : *array) {
      dimensions.push_back(getDimension(*val));
    }
    return Shape::DimensionSize(dimensions);
  }
  assert(false && "bad shape");

  return {-1};
}

static Shape getShape(const Type& type)
{
  llvm::SmallVector<Shape::DimensionSize, 3> dimensions;

  for (const auto& dim : type) {
    if (dim.hasExpression()) {
      const auto* expr = dim.getExpression();
      dimensions.push_back(getDimension(*expr));
    } else {
      dimensions.emplace_back(dim.getNumericSize());
    }
  }
  return Shape(dimensions);
}

void splitRaggedShape(
    llvm::ArrayRef<Shape::DimensionSize> dims,
    llvm::SmallVector<llvm::SmallVector<Shape::DimensionSize, 3>, 3>& results)
{
  //assuming results already of correct size
  for (const auto& it : dims)
    if (it.isRagged()) {
      const auto& ragged = it.asRagged();
      assert(results.size() == ragged.size());

      for (size_t index = 0U; index < results.size(); ++index)
        results[index].push_back(ragged[index]);
    } else {
      for (auto& res : results)
        res.push_back(it);
    }
}

static Type getTypeByShape(const Type& original, llvm::ArrayRef<Shape::DimensionSize> shape)
{
  Type new_type = original;

  llvm::SmallVector<ArrayDimension, 3> dims;
  for (const auto& d : shape)
    dims.emplace_back(d.asNum());

  new_type.setDimensions(dims);
  return new_type;
}

// e.g.
// id:'a' shape:[2][3]{4,5} :
// 'a[1]' [3][4]
// 'a[2]' [3][5]

// id:'b' shape:[2, {2,3}, {{4,5},2}]:
// 'b[1][1]' [4]
// 'b[1][2]' [5]
// 'b[2]' [3,2]
static void explodeMember(
    Member& member,
    const std::string& id,
    llvm::ArrayRef<Shape::DimensionSize> dims,
    bool& to_remove,
    llvm::SmallVectorImpl<std::unique_ptr<Member>>& to_add)
{
  if (!isRagged(dims))
    return;

  to_remove = true;

  for (size_t i = 1U; i < dims.size(); ++i) {

    if (isRagged(dims[i])) {

      auto slice = dims.slice(i);

      llvm::SmallVector<llvm::SmallVector<Shape::DimensionSize, 3>, 3> tmp(dims[i].asRagged().size());

      splitRaggedShape(slice, tmp);

      size_t index = 1U;
      for (const auto& splitted : tmp) {
        std::string member_id = id + "[" + std::to_string(index) + "]";

        if (isRagged(splitted)) {
          explodeMember(member, member_id, splitted, to_remove, to_add);
        } else {
          auto new_type = getTypeByShape(member.getType(), splitted);

          auto new_m = Member::build(
              member.getLocation(),
              member_id,
              new_type,
              member.getTypePrefix());

          to_add.emplace_back(std::move(new_m));
        }

        ++index;
      }
      return;
    }
  }
}
static void explodeMember(Member& member, bool& to_remove, llvm::SmallVectorImpl<std::unique_ptr<Member>>& to_add)
{
  if (isRagged(member)) {
    auto shape = getShape(member.getType());
    explodeMember(member, member.getName().str(), shape.dimensions(), to_remove, to_add);
  }
}

template<>
bool RaggedFlattener::run<Model>(Class& cls)
{
  SymbolTable::ScopeTy varScope(symbolTable);
  ShapeTable::ScopeTy shapeScope(shapeTable);

  auto* model = cls.get<Model>();

  for (const auto& innerClass : model->getInnerClasses())
    if (!run<Class>(*innerClass))
      return false;

  // Scalarize the members
  auto& members = model->getMembers_mut();

  // we need to keep the strings here since we are going to modify the ast, thus we can't use the references to the existing members names
  std::vector<std::string> dictionary(members.size());

  llvm::SmallVector<std::unique_ptr<Member>, 3> members_to_add;

  for (auto it = members.begin(); it != members.end();) {
    auto& member = **it;
    bool to_remove = false;

    if (isRagged(member)) {

      auto shape = getShape(member.getType());

      dictionary.push_back(member.getName().str());

      shapeTable.insert(dictionary[dictionary.size() - 1U], shape);
      explodeMember(member, to_remove, members_to_add);
    }

    if (to_remove) {
      it = members.erase(it);
    } else {
      it++;
    }
  }
  for (auto& it : members_to_add)
    members.emplace_back(std::move(it));

  // scalarize the equations and forEquations
  parentStack.push_back(model);
  for (auto& equationsBlock : model->getEquationsBlocks()) {
    forEquationsToAdd.clear();
    equationsToAdd.clear();

    parentStack.push_back(equationsBlock.get());

    for (auto& equation : equationsBlock->getEquations()) {
      parentStack.push_back(equation.get());
      if (!run(*equation)) {
        parentStack.pop_back();

        // when an equation is modified, it returns false in order to skip further modifications, the flag removeFlag it's used to specify it's not an error
        if (!removeFlag) return false;
        else
          removeFlag = true;
      }
    }
    for (auto& e : equationsToAdd)
      equationsBlock->add(std::move(e));

    for (auto& forEquation : equationsBlock->getForEquations()) {
      parentStack.push_back(forEquation.get());

      if (!run(*forEquation))
        return false;
      parentStack.pop_back();
    }
    for (auto& f : forEquationsToAdd)
      equationsBlock->add(std::move(f));

    parentStack.pop_back();
  }
  parentStack.pop_back();

  return true;
}

template<>
bool RaggedFlattener::run<Package>(Class& cls)
{
  SymbolTable::ScopeTy varScope(symbolTable);
  auto* package = cls.get<Package>();

  // Populate the symbol table
  symbolTable.insert(package->getName(), Symbol(cls));

  for (auto& innerClass : *package)
    symbolTable.insert(innerClass->getName(), Symbol(*innerClass));

  for (auto& innerClass : *package)
    if (!run<Class>(*innerClass))
      return false;

  return true;
}

template<>
bool RaggedFlattener::run<Record>(Class& cls)
{
  assert(cls.isa<Record>());
  auto* record = cls.get<Record>();

  symbolTable.insert(record->getName(), Symbol(cls));
  return true;
}

bool RaggedFlattener::run(Equation& equation)
{
  if (!run<Expression>(*equation.getLhsExpression()))
    return false;

  if (!run<Expression>(*equation.getRhsExpression()))
    return false;

  return true;
}

// an induction is ragged if it uses a variable from another induction, creating a non-rectangular loop
// returns the id of the other induction variable used
static llvm::Optional<std::string> inductionIsRagged(const Induction& induction)
{
  bool found = false;
  std::string varId;

  auto visitor = (ASTConstVisitor{
      [&](const Expression& e) {
        if (e.isa<ReferenceAccess>()) {
          found = true;
          varId = e.get<ReferenceAccess>()->getName();
        }
      },
      [](const auto& c) {}});

  auto isRagged = [&](const Expression* e) {
    visitor.visit(*e);
    return found;
  };

  if (isRagged(induction.getBegin()) || isRagged(induction.getEnd()) || isRagged(induction.getStep()))
    return varId;

  return {};
}

bool RaggedFlattener::run(ForEquation& forEquation)
{
  SymbolTable::ScopeTy varScope(symbolTable);

  for (auto& ind : forEquation.getInductions()) {
    symbolTable.insert(ind->getName(), Symbol(*ind));

    if (!run<Expression>(*ind->getBegin()))
      return false;

    if (!run<Expression>(*ind->getEnd()))
      return false;

    if (!run<Expression>(*ind->getStep()))
      return false;

    if (auto id = inductionIsRagged(*ind)) {

      // scalarizing the induction

      for (auto& ind : forEquation.getInductions()) {
        if (ind->getName() == *id) {
          int begin = getIntFromExpression(*ind->getBegin());
          int end = getIntFromExpression(*ind->getEnd());
          int step = getIntFromExpression(*ind->getStep());

          auto loc = ind->getLocation();
          auto name = ind->getName().str();

          for (int index = begin; index <= end; index += step) {

            llvm::SmallVector<std::unique_ptr<Induction>, 2> inductions;
            for (const auto& i : forEquation.getInductions()) {
              if (i->getName() != name) {
                inductions.push_back(std::move(i->clone()));
              }
            }

            auto new_e = forEquation.getEquation()->clone();

            TranslationTable::ScopeTy translationScope(translationTable);
            translationTable.insert(name, index);

            auto new_f = ForEquation::build(forEquation.getLocation(), std::move(inductions), std::move(new_e));

            // visit it in order to substitute the induction variable with the actual value
            if (!run(*new_f))
              return false;

            forEquationsToAdd.push_back(std::move(new_f));
          }
          // we end up generating more forEquations, we substitute the current one and add the other laters (adding during the iteration is bad)
          forEquation = *forEquationsToAdd[forEquationsToAdd.size() - 1];
          forEquationsToAdd.pop_back();
          return true;
        }
      }
    }
  }

  if (!run(*forEquation.getEquation()))
    return false;

  return true;
}

template<>
bool RaggedFlattener::run<Expression>(Expression& expression)
{
  return expression.visit([&](auto& obj) {
    using type = decltype(obj);
    using deref = typename std::remove_reference<type>::type;
    using deconst = typename std::remove_const<deref>::type;
    return run<deconst>(expression);
  });
}

template<>
bool RaggedFlattener::run<Array>(Expression& expression)
{
  return true;
}

template<>
bool RaggedFlattener::run<Call>(Expression& expression)
{
  return true;
}

template<>
bool RaggedFlattener::run<Constant>(Expression& expression)
{
  return true;
}

// compute how many subscripts need to be flattened
static size_t getNumFlattenedSubscripts(llvm::ArrayRef<Shape::DimensionSize> dims, llvm::ArrayRef<long> subscripts)
{
  assert(subscripts.size() <= dims.size());

  if (subscripts.empty())
    return 0U;

  assert(!dims[0].isRagged());

  size_t num = 0U;

  size_t max = std::min(subscripts.size() + 1U, dims.size());

  for (size_t i = 1U; i < max; ++i) {
    auto dim = dims[i];

    if (dim.isRagged()) {
      size_t index = subscripts[i - 1] - 1U;// since modelica use 1-indexed array

      llvm::SmallVector<llvm::SmallVector<Shape::DimensionSize, 3>, 3> tmp(dim.asRagged().size());

      //TODO: optimize this, no need to split the whole shape, just to navigate to the correct part
      splitRaggedShape(dims.slice(1), tmp);
      auto new_dims = tmp[index];
      return 1 + getNumFlattenedSubscripts(new_dims, subscripts.slice(i));
    }
  }

  return num;
}

//returns <begin,end,step>
static llvm::Optional<std::tuple<int, int, int>> subscriptNeedIteration(const Expression& expression)
{
  if (auto p = expression.dyn_get<Constant>()) {
    if (p->as<BuiltInType::Integer>() == -1) {
      return std::make_tuple<int, int, int>(1, -1, 1);
    }
  }
  if (auto p = expression.dyn_get<Operation>()) {

    if (p->getOperationKind() == OperationKind::range) {
      auto args = p->getArguments();
      assert(args.size() >= 2);
      if (args.size() == 2) return std::make_tuple<int, int, int>(getIntFromExpression(*args[0]), getIntFromExpression(*args[1]), 1);
      return std::make_tuple<int, int, int>(getIntFromExpression(*args[0]), getIntFromExpression(*args[2]), getIntFromExpression(*args[1]));
    }
  }
  return llvm::None;
}

template<>
bool RaggedFlattener::run<Operation>(Expression& expression)
{
  auto* operation = expression.get<Operation>();
  auto type = expression.getType();

  for (auto& arg : operation->getArguments())
    if (!run<Expression>(*arg))
      return false;

  if (operation->getOperationKind() == OperationKind::subscription) {
    auto* lhs = operation->getArg(0);
    auto lhs_type = lhs->getType();

    // handle the scalarization of ranges equations, e.g:
    // 'c.x'[:,1] = 'c.p'[:];
    // becomes:
    // // 'c.x[1]'[1] = 'c.p'[1];
    // // 'c.x[2]'[1] = 'c.p'[2];
    // // 'c.x[3]'[1] = 'c.p'[3];
    if (range_index) {
      for (auto& s : operation->getArguments().slice(1)) {
        if (auto p = s->dyn_get<Constant>()) {
          if (p->as<BuiltInType::Integer>() == -1) {
            *s = *Expression::constant(p->getLocation(), p->getType(), *range_index);
          }
        }
      }
    }

    if (auto reference = lhs->dyn_get<ReferenceAccess>()) {
      auto name = reference->getName().str();

      if (shapeTable.count(name)) {
        // if its a subscript to ragged member, try to scalarize it
        auto shape = shapeTable.lookup(name);

        auto subscripts_expr = operation->getArguments().slice(1);
        std::vector<long> subscripts;

        size_t subscript_index = 0;
        for (const auto& s : subscripts_expr) {

          if (auto loop = subscriptNeedIteration(*s)) {

            int begin = std::get<0>(*loop);
            int end = std::get<1>(*loop);
            int step = std::get<2>(*loop);

            if (end == -1) {
              //todo handle other subscript_index
              assert(subscript_index == 0);
              end = shape.dimensions()[subscript_index].asNum();
            }

            auto m = getModel();
            assert(m);
            auto e = static_cast<Equation*>(parentStack[parentStack.size() - 1]);
            assert(e);

            for (int index = begin; index <= end; index += step) {
              // set the range_index in order to translating the ':'(-1) with the scalarization value when visiting
              range_index = {index};

              auto new_e = e->clone();
              run(*new_e);

              range_index = llvm::None;

              equationsToAdd.push_back(std::move(new_e));
            }
            // we end up generating more Equations, we substitute the current one and add the other laters (adding during the iteration is bad)
            *e = *equationsToAdd[equationsToAdd.size() - 1U];
            equationsToAdd.pop_back();

            // we want to skip further modification that can mess up the AST, so we return false and set the removeFlag
            // (problems can arise when we are visiting the lhs part of the equation and the rhs would be next)
            // (the equation has already been visited above in the loop anyway)
            removeFlag = true;
            return false;
          } else if (auto p = s->dyn_get<Constant>()) {
            subscripts.push_back(p->as<BuiltInType::Integer>());
          } else
            break;

          ++subscript_index;
        }

        // simple scalarazation, if c.x is ragged, c.x[1] becomes -> 'c.x[1]'

        auto n = getNumFlattenedSubscripts(shape.dimensions(), subscripts);

        for (size_t i = 0; i < n; ++i) {
          name += '[' + std::to_string(subscripts[i]) + ']';
          operation->removeArg(1);
        }

        reference->setName(name);
      }
    } else if (auto array = lhs->dyn_get<Array>()) {
      // check if it is possible to apply a constant folding on the array
      // since static array are usually used in ragged loop
      auto firstSubscript = operation->getArg(1);

      if (auto p = firstSubscript->dyn_get<Constant>()) {

        int index = p->as<BuiltInType::Integer>() - 1;// -1 since modelica uses 1-indexed array
        assert(index >= 0 && (size_t) index < array->size());

        if (operation->size() == 2) {
          // if it's just one subscript, substitute the operation with the correct array value
          expression = *(*array)[index];
          return true;
        } else {
          // otherwise generate a new operation with the new expression and the remaining subscripts

          llvm::SmallVector<std::unique_ptr<Expression>, 2> args;
          args.emplace_back(std::move((*array)[index]));

          auto subscripts_expr = operation->getArguments().slice(2);
          for (auto& s : subscripts_expr) {
            args.emplace_back(std::move(s));
          }
          expression = *Expression::operation(operation->getLocation(), operation->getType().subscript(1), OperationKind::subscription, args);

          // visit it to folding it recursively (this way multidimensional constant array are correctly handled)
          return run<Expression>(expression);
        }
      }
    }

    if (operation->argumentsCount() == 1) {
      expression = *lhs;
      return true;
    }
  }

  return true;
}

template<>
bool RaggedFlattener::run<ReferenceAccess>(Expression& expression)
{
  auto* reference = expression.get<ReferenceAccess>();
  auto name = reference->getName();

  if (translationTable.count(name)) {
    // replace reference with the current scalarization value
    expression = *Expression::constant(expression.getLocation(), makeType<BuiltInType::Integer>(), translationTable.lookup(name));
  }

  return true;
}

template<>
bool RaggedFlattener::run<Tuple>(Expression& expression)
{
  auto* tuple = expression.get<Tuple>();

  for (auto& exp : *tuple)
    if (!run<Expression>(*exp))
      return false;

  return true;
}

bool RaggedFlattener::run(Member& member)
{
  return true;
}

template<>
bool RaggedFlattener::run<Statement>(Statement& statement)
{
  return statement.visit([&](auto& obj) {
    using type = decltype(obj);
    using deref = typename std::remove_reference<type>::type;
    using deconst = typename std::remove_const<deref>::type;
    return run<deconst>(statement);
  });
}

template<>
bool RaggedFlattener::run<AssignmentStatement>(Statement& statement)
{
  auto* assignmentStatement = statement.get<AssignmentStatement>();
  auto* expression = assignmentStatement->getExpression();

  if (!run<Expression>(*expression))
    return false;

  return true;
}

template<>
bool RaggedFlattener::run<BreakStatement>(Statement& statement)
{
  return true;
}

template<>
bool RaggedFlattener::run<ForStatement>(Statement& statement)
{
  return true;
}

bool RaggedFlattener::run(Induction& induction)
{
  if (!run<Expression>(*induction.getBegin())) {
    return false;
  }

  if (!run<Expression>(*induction.getEnd())) {
    return true;
  }

  if (!run<Expression>(*induction.getStep())) {
    return true;
  }

  return true;
}

template<>
bool RaggedFlattener::run<IfStatement>(Statement& statement)
{
  return true;
}

template<>
bool RaggedFlattener::run<ReturnStatement>(Statement& statement)
{
  return true;
}

template<>
bool RaggedFlattener::run<WhenStatement>(Statement& statement)
{
  return true;
}

template<>
bool RaggedFlattener::run<WhileStatement>(Statement& statement)
{
  return true;
}

bool RaggedFlattener::run(Algorithm& algorithm)
{
  return true;
}

std::unique_ptr<Pass> marco::ast::createRaggedFlatteningPass(diagnostic::DiagnosticEngine& diagnostics)
{
  return std::make_unique<RaggedFlattener>(diagnostics);
}
