#include "marco/Dialect/Modelica/Transforms/Instantiation/InstanceTree.h"
#include "mlir/IR/BuiltinOps.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::instantiation;

//===---------------------------------------------------------------------===//
// Modifier
//===---------------------------------------------------------------------===//

namespace mlir::modelica::instantiation {
const ModifierValue *Modifier::getValue() const {
  if (!value || value->op.getIsBreak()) {
    return nullptr;
  }

  return &*value;
}

const Modifier *Modifier::getChild(llvm::StringRef name) const {
  auto it = children.find(name.str());
  return it == children.end() ? nullptr : &it->second;
}

Modifier Modifier::merge(const Modifier &outer, const Modifier &inner) {
  Modifier result = inner;

  if (outer.value) {
    result.value = outer.value;
    result.isEach = outer.isEach;
  }

  for (const auto &[name, child] : outer.children) {
    auto it = result.children.find(name);

    if (it == result.children.end()) {
      result.children.emplace(name, child);
    } else {
      it->second = merge(child, it->second);
    }
  }

  for (const auto &[name, redeclaration] : outer.redeclarations) {
    result.redeclarations[name] = redeclaration;
  }

  return result;
}

void Modifier::split(llvm::ArrayRef<unsigned> positions) {
  if (isEach) {
    return;
  }

  if (value) {
    value->splitPositions.append(positions.begin(), positions.end());
  }

  for (auto &child : llvm::make_second_range(children)) {
    child.split(positions);
  }
}

Modifier Modifier::parse(mlir::Region &region, Scope *scope) {
  Modifier result;

  if (region.empty()) {
    return result;
  }

  for (mlir::Operation &op : region.front()) {
    if (auto modifyOp = mlir::dyn_cast<ModifyOp>(op)) {
      // Composite names, such as 'a.b.c = 1', modify nested elements.
      Modifier *target =
          &result.children[modifyOp.getName().getRootReference().str()];

      for (mlir::FlatSymbolRefAttr nested :
           modifyOp.getName().getNestedReferences()) {
        target = &target->children[nested.getValue().str()];
      }

      Modifier nestedModifier = parse(modifyOp.getModificationRegion(), scope);
      nestedModifier.isEach = modifyOp.getIsEach();
      *target = merge(nestedModifier, *target);
      continue;
    }

    if (auto bindOp = mlir::dyn_cast<BindOp>(op)) {
      result.value = ModifierValue{bindOp, scope, {}};
      continue;
    }

    if (mlir::isa<ClassOp, EnumerationOp, ComponentOp>(op)) {
      auto name = op.getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName());

      Modifier &child = result.children[name.str()];
      child.isEach = static_cast<bool>(op.getAttr("isEach"));
      result.redeclarations[name.str()] = Redeclaration{&op, scope};
    }
  }

  return result;
}

//===---------------------------------------------------------------------===//
// Element
//===---------------------------------------------------------------------===//

llvm::StringRef Element::getName() const {
  if (!op) {
    switch (predefined) {
    case PredefinedClass::Real:
      return "Real";
    case PredefinedClass::Integer:
      return "Integer";
    case PredefinedClass::Boolean:
      return "Boolean";
    case PredefinedClass::String:
      return "String";
    case PredefinedClass::StateSelect:
      return "StateSelect";
    case PredefinedClass::AssertionLevel:
      return "AssertionLevel";
    case PredefinedClass::ExternalObject:
      return "ExternalObject";
    case PredefinedClass::Connections:
      return "Connections";
    case PredefinedClass::Clock:
      return "Clock";
    case PredefinedClass::None:
      break;
    }

    return "";
  }

  return op->getAttrOfType<mlir::StringAttr>(
               mlir::SymbolTable::getSymbolAttrName())
      .getValue();
}

//===---------------------------------------------------------------------===//
// Scope
//===---------------------------------------------------------------------===//

bool Scope::isRoot() const { return op && mlir::isa<mlir::ModuleOp>(op); }

bool Scope::isEncapsulated() const {
  if (auto classOp = mlir::dyn_cast_or_null<ClassOp>(op)) {
    return classOp.getIsEncapsulated();
  }

  return false;
}

ClassKind Scope::getKind() const {
  if (auto classOp = mlir::dyn_cast_or_null<ClassOp>(op)) {
    return classOp.getKind();
  }

  if (isRoot()) {
    return ClassKind::package;
  }

  return ClassKind::type;
}

mlir::Location Scope::getLoc() const {
  if (op) {
    return op->getLoc();
  }

  return parent ? parent->getLoc()
                : mlir::UnknownLoc::get(modifier.value
                                            ? modifier.value->op.getContext()
                                            : nullptr);
}

//===---------------------------------------------------------------------===//
// Component
//===---------------------------------------------------------------------===//

ComponentOp Component::getDeclaration() const {
  return element ? mlir::dyn_cast<ComponentOp>(element->op) : nullptr;
}

bool Component::isOuter() const {
  ComponentOp op = getDeclaration();
  return op && op.getIsOuter() && !op.getIsInner() &&
         !element->isImplicitInner;
}

bool Component::isInner() const {
  ComponentOp op = getDeclaration();
  return op && (op.getIsInner() || element->isImplicitInner);
}

std::string Component::getPath() const {
  if (!parent || parent->isRoot()) {
    return name;
  }

  return parent->getPath() + "." + name;
}
} // namespace mlir::modelica::instantiation
