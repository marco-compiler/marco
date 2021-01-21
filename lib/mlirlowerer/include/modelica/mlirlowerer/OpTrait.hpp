#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/Type.hpp>

namespace modelica
{
	/**
	 * This class verifies that all operands of the specified operation have a
	 * signless integer, float or index type, a vector thereof, or a tensor
	 * thereof.
	 */
	template<typename ConcreteType>
	class OperandsAreSignlessIntegerOrFloatLike
			: public mlir::OpTrait::
						TraitBase<ConcreteType, OperandsAreSignlessIntegerOrFloatLike>
	{
		public:
		static mlir::LogicalResult verifyTrait(mlir::Operation* op)
		{
			if (failed(mlir::OpTrait::impl::verifyOperandsAreSignlessIntegerLike(op)))
				return mlir::OpTrait::impl::verifyOperandsAreFloatLike(op);

			return mlir::success();
		}
	};
}