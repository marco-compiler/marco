#include "marco/Codegen/Lowering/Reference.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  Reference::Reference()
      : builder(nullptr),
        value(nullptr),
        reader(nullptr)
  {
  }

  Reference::Reference(mlir::OpBuilder* builder,
                       mlir::Value value,
                       std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader,
                       std::function<void(mlir::OpBuilder*, Reference&, mlir::Value)> writer)
      : builder(builder),
        value(std::move(value)),
        reader(std::move(reader)),
        writer(std::move(writer))
  {
  }

  mlir::Value Reference::operator*() const
  {
    return reader(builder, value);
  }

  mlir::Value Reference::getReference() const
  {
    assert(value != nullptr);
    return value;
  }

  void Reference::set(mlir::Value v)
  {
    writer(builder, *this, v);
  }

  Reference Reference::ssa(mlir::OpBuilder* builder, mlir::Value value)
  {
    return Reference(
        builder, value,
        [](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value {
          return value;
        },
        [](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
          llvm_unreachable("Can't assign value to SSA operand");
        });
  }

  Reference Reference::memory(mlir::OpBuilder* builder, mlir::Value value)
  {
    return Reference(
        builder, value,
        [](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value {
          auto arrayType = value.getType().cast<ArrayType>();

          // We can load the value only if it's a pointer to a scalar.
          // Otherwise, return the array.

          if (arrayType.getShape().empty()) {
            return builder->create<LoadOp>(value.getLoc(), value);
          }

          return value;
        },
        [&](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
          assert(destination.value.getType().isa<ArrayType>());
          builder->create<AssignmentOp>(value.getLoc(), destination.getReference(), value);
        });
  }

  Reference Reference::member(mlir::OpBuilder* builder, mlir::Value value)
  {
    return Reference(
        builder, value,
        [](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value {
          auto memberType = value.getType().cast<MemberType>();
          return builder->create<MemberLoadOp>(value.getLoc(), memberType.unwrap(), value);
        },
        [](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
          builder->create<MemberStoreOp>(value.getLoc(), destination.value, value);
        });
  }

  Reference Reference::time(mlir::OpBuilder* builder)
  {
    return Reference(
        builder, nullptr,
        [](mlir::OpBuilder* builder, mlir::Value v) -> mlir::Value {
          return builder->create<TimeOp>(builder->getUnknownLoc());
        },
        [](mlir::OpBuilder* builder, Reference& destination, mlir::Value v) {
          llvm_unreachable("Can't write into the 'time' variable");
        });
  }
}
