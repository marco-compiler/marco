#include "marco/Codegen/Lowering/BaseModelica/Reference.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::marco::codegen::lowering::bmodelica;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
class Reference::Impl {
public:
  Impl(mlir::OpBuilder &builder, mlir::Location loc)
      : builder(&builder), loc(std::move(loc)) {}

  virtual ~Impl() = default;

  virtual std::unique_ptr<Reference::Impl> clone() = 0;

  mlir::Location getLoc() const { return loc; }

  virtual mlir::Operation *getOperation() const = 0;

  virtual mlir::Value getReference() const = 0;

  virtual mlir::Value get(mlir::Location loc) const = 0;

  virtual void set(mlir::Location loc, mlir::ValueRange indices,
                   mlir::Value value) = 0;

protected:
  mlir::OpBuilder *builder;
  mlir::Location loc;
};
} // namespace marco::codegen::lowering::bmodelica

namespace {
class SSAReference : public Reference::Impl {
public:
  SSAReference(mlir::OpBuilder &builder, mlir::Value value)
      : Reference::Impl(builder, value.getLoc()), reference(value) {}

  std::unique_ptr<Reference::Impl> clone() override {
    return std::make_unique<SSAReference>(*this);
  }

  mlir::Value getReference() const override { return reference; }

  mlir::Operation *getOperation() const override {
    return reference.getDefiningOp();
  }

  mlir::Value get(mlir::Location loc) const override { return reference; }

  void set(mlir::Location loc, mlir::ValueRange indices,
           mlir::Value value) override {
    llvm_unreachable("Can't assign value to SSA operand");
  }

private:
  mlir::Value reference;
};

class TensorReference : public Reference::Impl {
public:
  TensorReference(mlir::OpBuilder &builder, mlir::Value value)
      : Reference::Impl(builder, value.getLoc()), reference(value) {
    assert(mlir::isa<mlir::TensorType>(reference.getType()));
  }

  std::unique_ptr<Reference::Impl> clone() override {
    return std::make_unique<TensorReference>(*this);
  }

  mlir::Operation *getOperation() const override {
    return reference.getDefiningOp();
  }

  mlir::Value get(mlir::Location loc) const override {
    auto tensorType = mlir::cast<mlir::TensorType>(reference.getType());

    if (tensorType.getShape().empty()) {
      return builder->create<TensorExtractOp>(loc, reference);
    }

    return reference;
  }

  mlir::Value getReference() const override { return reference; }

  void set(mlir::Location loc, mlir::ValueRange indices,
           mlir::Value value) override {
    llvm_unreachable("Not implemented");
  }

private:
  mlir::Value reference;
};

class VariableReference : public Reference::Impl {
public:
  VariableReference(mlir::OpBuilder &builder, VariableOp variableOp)
      : Reference::Impl(builder, variableOp.getLoc()), variableOp(variableOp) {}

  std::unique_ptr<Reference::Impl> clone() override {
    return std::make_unique<VariableReference>(*this);
  }

  mlir::Operation *getOperation() const override { return variableOp; }

  mlir::Value get(mlir::Location loc) const override {
    return builder->create<VariableGetOp>(loc, variableOp);
  }

  mlir::Value getReference() const override {
    llvm_unreachable("Variables have no SSA value");
  }

  void set(mlir::Location loc, mlir::ValueRange indices,
           mlir::Value value) override {
    builder->create<VariableSetOp>(loc, variableOp, indices, value);
  }

private:
  VariableOp variableOp;
};

class ComponentReference : public Reference::Impl {
public:
  ComponentReference(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value parent, mlir::Type componentType,
                     llvm::StringRef componentName)
      : Reference::Impl(builder, loc), parent(parent),
        componentType(componentType), componentName(componentName.str()) {}

  std::unique_ptr<Reference::Impl> clone() override {
    return std::make_unique<ComponentReference>(*this);
  }

  mlir::Operation *getOperation() const override {
    llvm_unreachable("Component references have no defining operation");
    return nullptr;
  }

  mlir::Value get(mlir::Location loc) const override {
    return builder->create<ComponentGetOp>(loc, componentType, parent,
                                           componentName);
  }

  mlir::Value getReference() const override {
    llvm_unreachable("Variable components have no SSA value");
    return {};
  }

  void set(mlir::Location loc, mlir::ValueRange indices,
           mlir::Value value) override {
    llvm_unreachable("Records are not supposed to be modified");
  }

private:
  mlir::Value parent;
  mlir::Type componentType;
  std::string componentName;
};

class TimeReference : public Reference::Impl {
public:
  TimeReference(mlir::OpBuilder &builder, mlir::Location loc)
      : Reference::Impl(builder, loc) {}

  std::unique_ptr<Reference::Impl> clone() override {
    return std::make_unique<TimeReference>(*this);
  }

  mlir::Operation *getOperation() const override {
    llvm_unreachable("The 'time' reference has no defining operation");
    return nullptr;
  }

  mlir::Value get(mlir::Location loc) const override {
    return builder->create<TimeOp>(loc);
  }

  mlir::Value getReference() const override {
    llvm_unreachable("No reference for the 'time' variable");
    return {};
  }

  void set(mlir::Location loc, mlir::ValueRange indices,
           mlir::Value value) override {
    llvm_unreachable("Can't write into the 'time' variable");
  }
};
} // namespace

namespace marco::codegen::lowering::bmodelica {
Reference::Reference() : impl(nullptr) {}

Reference::Reference(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

Reference::~Reference() = default;

Reference::Reference(const Reference &other) {
  if (other.impl) {
    impl = other.impl->clone();
  }
}

Reference &Reference::operator=(const Reference &other) {
  Reference result(other);
  swap(*this, result);
  return *this;
}

Reference &Reference::operator=(Reference &&other) = default;

void swap(Reference &first, Reference &second) {
  using std::swap;
  swap(first.impl, second.impl);
}

Reference Reference::ssa(mlir::OpBuilder &builder, mlir::Value value) {
  return Reference(std::make_unique<SSAReference>(builder, value));
}

Reference Reference::tensor(mlir::OpBuilder &builder, mlir::Value value) {
  return Reference(std::make_unique<TensorReference>(builder, value));
}

Reference Reference::variable(mlir::OpBuilder &builder, VariableOp variableOp) {
  return Reference(std::make_unique<VariableReference>(builder, variableOp));
}

Reference Reference::component(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value parent, mlir::Type componentType,
                               llvm::StringRef componentName) {
  return Reference(std::make_unique<ComponentReference>(
      builder, loc, parent, componentType, componentName));
}

Reference Reference::time(mlir::OpBuilder &builder, mlir::Location loc) {
  return Reference(std::make_unique<TimeReference>(builder, loc));
}

mlir::Value Reference::get(mlir::Location loc) const { return impl->get(loc); }

mlir::Location Reference::getLoc() const { return impl->getLoc(); }

mlir::Operation *Reference::getOperation() const {
  return impl->getOperation();
}

mlir::Value Reference::getReference() const { return impl->getReference(); }

void Reference::set(mlir::Location loc, mlir::ValueRange indices,
                    mlir::Value value) {
  impl->set(loc, indices, value);
}
} // namespace marco::codegen::lowering::bmodelica
