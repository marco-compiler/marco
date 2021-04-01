#pragma once

#include <mlir/IR/Value.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

#include "Model.h"

namespace modelica::codegen::model
{
	class Expression;

	class ModelBuilder
	{
		public:
		ModelBuilder(Model& model);

		void lower(EquationOp equation);
		void lower(ForEquationOp forEquation);
		Expression lower(mlir::Value value);

		private:
		Model& model;
	};

}
