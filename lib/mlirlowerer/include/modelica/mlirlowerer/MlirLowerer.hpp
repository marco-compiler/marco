#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/utils/SourceRange.hpp>
#include <mlir/InitAllDialects.h>

namespace modelica
{
	class MlirLowerer
	{
		public:
		explicit MlirLowerer(mlir::MLIRContext &context);

		mlir::FuncOp lower(const modelica::ClassContainer& cls);
		mlir::FuncOp lower(const modelica::Class& cls);
		mlir::FuncOp lower(const modelica::Function& function);
		mlir::Type lower(const modelica::Type& type);
		mlir::Type lower(const modelica::BuiltInType& type);
		mlir::Type lower(const modelica::UserDefinedType& type);
		mlir::LogicalResult lower(const modelica::Algorithm& algorithm);
		mlir::LogicalResult lower(const modelica::Statement& statement);
		mlir::LogicalResult lower(const modelica::AssignmentStatement& statement);
		mlir::LogicalResult lower(const modelica::IfStatement& statement);
		mlir::LogicalResult lower(const modelica::ForStatement& statement);
		mlir::LogicalResult lower(const modelica::WhileStatement& statement);
		mlir::LogicalResult lower(const modelica::WhenStatement& statement);
		mlir::LogicalResult lower(const modelica::BreakStatement& statement);
		mlir::LogicalResult lower(const modelica::ReturnStatement& statement);
		mlir::Value lower(const modelica::Expression& expression);
		mlir::Value lower(const modelica::Operation& operation);
		mlir::Value lower(const modelica::Constant& constant);
		mlir::Value lower(const modelica::ReferenceAccess& reference);
		mlir::Value lower(const modelica::Call& call);
		mlir::Value lower(const modelica::Tuple& tuple);

		private:
		/// The builder is a helper class to create IR inside a function. The
		/// builder is stateful, in particular it keeps an "insertion point":
		/// this is where the next operations will be introduced.
		mlir::OpBuilder builder;

		/// The symbol table maps a variable name to a value in the current scope.
		/// Entering a function creates a new scope, and the function arguments
		/// are added to the mapping. When the processing of a function is
		/// terminated, the scope is destroyed and the mappings created in this
		/// scope are dropped.
		llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);

		/**
		 * Declare a variable in the current scope.
		 *
		 * @param var 	variable name
		 * @param value value
		 * @return success if the variable wasn't declared yet
		 */
		mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value);

		mlir::Type constantToType(const Constant& constant)
		{
			if (constant.isA<BuiltInType::Boolean>())
				return builder.getI1Type();

			if (constant.isA<BuiltInType::Integer>())
				return builder.getI32Type();

			if (constant.isA<BuiltInType::Float>())
				return builder.getF32Type();

			assert(false && "Unreachable");
		}

		template<typename T>
		constexpr mlir::Attribute getAttribute(const T& value)
		{
			constexpr BuiltInType type = typeToFrontendType<T>();

			if constexpr (type == BuiltInType::Boolean)
				return builder.getBoolAttr(value);
			else if constexpr (type == BuiltInType::Integer)
				return builder.getI32IntegerAttr(value);
			else if constexpr (type == BuiltInType::Float)
				return builder.getF32FloatAttr(value);

			assert(false && "Unreachable");
		}
	};
}
