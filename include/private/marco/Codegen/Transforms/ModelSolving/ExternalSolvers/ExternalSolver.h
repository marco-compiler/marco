#ifndef MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVERS_EXTERNALSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVERS_EXTERNALSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class IDASolver;

  class ExternalSolver
  {
    public:
      ExternalSolver(mlir::TypeConverter* typeConverter);

      virtual ~ExternalSolver();

      virtual bool isEnabled() const = 0;

      virtual void setEnabled(bool status) = 0;

      virtual bool containsEquation(ScheduledEquation* equation) const = 0;

      virtual mlir::Type getRuntimeDataType(mlir::MLIRContext* context) = 0;

      virtual mlir::LogicalResult processInitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::func::FuncOp initFunction,
          mlir::ValueRange variables,
          const Model<ScheduledEquationsBlock>& model) = 0;

      virtual mlir::LogicalResult processDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::func::FuncOp deinitFunction) = 0;

      virtual mlir::LogicalResult processUpdateStatesFunction(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr,
          mlir::func::FuncOp updateStatesFunction,
          mlir::ValueRange variables) = 0;

      virtual bool hasTimeOwnership() const = 0;

      virtual mlir::Value getCurrentTime(
          mlir::OpBuilder& builder,
          mlir::Value runtimeDataPtr) = 0;

    protected:
      mlir::TypeConverter* getTypeConverter();

    private:
      mlir::TypeConverter* typeConverter;
  };

  class ExternalSolvers
  {
    private:
      using Container = std::vector<std::unique_ptr<ExternalSolver>>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      void addSolver(std::unique_ptr<ExternalSolver> solver);

      bool containsEquation(ScheduledEquation* equation) const;

      mlir::Value getCurrentTime(mlir::OpBuilder& builder, mlir::ValueRange runtimeDataPtrs) const;

      size_t size() const;

      iterator begin();
      const_iterator begin() const;

      iterator end();
      const_iterator end() const;

    public:
      Container solvers;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXTERNALSOLVERS_EXTERNALSOLVER_H
