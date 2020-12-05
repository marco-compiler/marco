#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>

namespace modelica
{
	class ConstantOpOld: public mlir::Op<
													 ConstantOpOld, mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult> {

		public:
		/// Inherit the constructors from the base Op class.
		using Op::Op;

		/// Provide the unique name for this operation. MLIR will use this to register
		/// the operation and uniquely identify it throughout the system.
		static llvm::StringRef getOperationName();

		/// Return the value of the constant by fetching it from the attribute.
		mlir::Attribute getValue();

		/// Operations can provide additional verification beyond the traits they
		/// define. Here we will ensure that the specific invariants of the constant
		/// operation are upheld, for example the result type must be of TensorType.
		mlir::LogicalResult verify();

		/// Provide an interface to build this operation from a set of input values.
		/// This interface is used by the builder to allow for easily generating
		/// instances of this operation:
		///   mlir::OpBuilder::create<ConstantOp>(...)
		/// This method populates the given 'state' that MLIR uses to create
		/// operations. This state is a collection of all of the discrete elements
		/// that an operation may contain.
		/// Build a constant with the given return type and 'value' attribute.
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type, mlir::Attribute value);
	};
}
