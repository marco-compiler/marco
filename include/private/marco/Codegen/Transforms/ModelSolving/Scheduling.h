#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCHEDULING_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCHEDULING_H

#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Modeling/Scheduling.h"
#include <memory>

namespace marco::codegen
{
  class ScheduledEquation : public Equation
  {
    public:
      ScheduledEquation(
          std::unique_ptr<MatchedEquation> equation,
          modeling::MultidimensionalRange scheduledIndexes,
          modeling::scheduling::Direction schedulingDirection);

      ScheduledEquation(const ScheduledEquation& other);

      ~ScheduledEquation();

      ScheduledEquation& operator=(const ScheduledEquation& other);
      ScheduledEquation& operator=(ScheduledEquation&& other);

      friend void swap(ScheduledEquation& first, ScheduledEquation& second);

      std::unique_ptr<Equation> clone() const override;

      /// @name Forwarded methods
      /// {

      mlir::modelica::EquationOp cloneIR() const override;

      void eraseIR() override;

      void dumpIR() const override;

      void dumpIR(llvm::raw_ostream& os) const override;

      mlir::modelica::EquationOp getOperation() const override;

      Variables getVariables() const override;

      void setVariables(Variables variables) override;

      std::vector<Access> getAccesses() const override;

      ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const override;

      mlir::Value getValueAtPath(const EquationPath& path) const override;

      Access getAccessAtPath(const EquationPath& path) const override;

      std::vector<Access> getReads() const;

      Access getWrite() const;

      mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder, const EquationPath& path) override;

      std::unique_ptr<Equation> cloneIRAndExplicitate(
          mlir::OpBuilder& builder, const EquationPath& path) const override;

      std::unique_ptr<Equation> cloneIRAndExplicitate(mlir::OpBuilder& builder) const;

      std::vector<mlir::Value> getInductionVariables() const override;

      mlir::LogicalResult replaceInto(
          mlir::OpBuilder& builder,
          Equation& destination,
          const modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath) const override;

      mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          modeling::scheduling::Direction iterationDirection) const override;

      /// }
      /// @name Modified methods
      /// {

      size_t getNumOfIterationVars() const override;

      modeling::MultidimensionalRange getIterationRanges() const override;

      /// }

      /// Get the direction to be used to update the iteration variables.
      ::marco::modeling::scheduling::Direction getSchedulingDirection() const;

    private:
      std::unique_ptr<MatchedEquation> equation;
      modeling::MultidimensionalRange scheduledIndexes;
      modeling::scheduling::Direction schedulingDirection;
  };

  // Specialize the container for equations in case of scheduled ones, in order to
  // force them to be ordered.
  template<>
  class Equations<ScheduledEquation>
  {
    private:
      using Impl = impl::Equations<ScheduledEquation, std::vector>;

    public:
      using iterator = typename Impl::iterator;
      using const_iterator = typename Impl::const_iterator;

      Equations() : impl(std::make_shared<Impl>())
      {
      }

      /// @name Forwarded methods
      /// {

      void push_back(std::unique_ptr<ScheduledEquation> equation)
      {
        impl->add(std::move(equation));
      }

      size_t size() const
      {
        return impl->size();
      }

      std::unique_ptr<ScheduledEquation>& operator[](size_t index)
      {
        return (*impl)[index];
      }

      const std::unique_ptr<ScheduledEquation>& operator[](size_t index) const
      {
        return (*impl)[index];
      }

      iterator begin()
      {
        return impl->begin();
      }

      const_iterator begin() const
      {
        return impl->begin();
      }

      iterator end()
      {
        return impl->end();
      }

      const_iterator end() const
      {
        return impl->end();
      }

      void setVariables(Variables variables)
      {
        impl->setVariables(std::move(variables));
      }

      /// }

    private:
      std::shared_ptr<Impl> impl;
  };

  class ScheduledEquationsBlock
  {
    private:
      using Container = Equations<ScheduledEquation>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      ScheduledEquationsBlock(Equations<ScheduledEquation> equations, bool hasCycle)
        : equations(std::move(equations)),
          cycle(std::move(hasCycle))
      {
      }

      bool hasCycle() const
      {
        return cycle;
      }

      /// @name Forwarded methods
      /// {

      std::unique_ptr<ScheduledEquation>& operator[](size_t index)
      {
        return equations[index];
      }

      const std::unique_ptr<ScheduledEquation>& operator[](size_t index) const
      {
        return equations[index];
      }

      iterator begin()
      {
        return equations.begin();
      }

      const_iterator begin() const
      {
        return equations.begin();
      }

      iterator end()
      {
        return equations.end();
      }

      const_iterator end() const
      {
        return equations.end();
      }

      void setVariables(Variables variables)
      {
        equations.setVariables(std::move(variables));
      }

      /// }

    private:
      Equations<ScheduledEquation> equations;
      bool cycle;
  };

  namespace impl
  {
    class ScheduledEquationsBlocks
    {
      private:
        using Container = std::vector<std::unique_ptr<ScheduledEquationsBlock>>;

      public:
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;

        std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index)
        {
          assert(index < scheduledBlocks.size());
          return scheduledBlocks[index];
        }

        const std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index) const
        {
          assert(index < scheduledBlocks.size());
          return scheduledBlocks[index];
        }

        size_t size() const
        {
          return scheduledBlocks.size();
        }

        void append(std::unique_ptr<ScheduledEquationsBlock> block)
        {
          scheduledBlocks.push_back(std::move(block));
        }

        /// @name Iterators
        /// {

        iterator begin()
        {
          return scheduledBlocks.begin();
        }

        const_iterator begin() const
        {
          return scheduledBlocks.begin();
        }

        iterator end()
        {
          return scheduledBlocks.end();
        }

        const_iterator end() const
        {
          return scheduledBlocks.end();
        }

        /// }

      private:
        Container scheduledBlocks;
    };
  }

  template<>
  class Equations<ScheduledEquationsBlock>
  {
    private:
      using Impl = impl::Equations<ScheduledEquationsBlock, std::vector>;

    public:
      using iterator = typename Impl::iterator;
      using const_iterator = typename Impl::const_iterator;

      Equations() : impl(std::make_shared<Impl>())
      {
      }

      /// @name Forwarded methods
      /// {

      void push_back(std::unique_ptr<ScheduledEquationsBlock> equation)
      {
        impl->add(std::move(equation));
      }

      size_t size() const
      {
        return impl->size();
      }

      std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index)
      {
        return (*impl)[index];
      }

      const std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index) const
      {
        return (*impl)[index];
      }

      iterator begin()
      {
        return impl->begin();
      }

      const_iterator begin() const
      {
        return impl->begin();
      }

      iterator end()
      {
        return impl->end();
      }

      const_iterator end() const
      {
        return impl->end();
      }

      void setVariables(Variables variables)
      {
        impl->setVariables(std::move(variables));
      }

      /// }

    private:
      std::shared_ptr<Impl> impl;
  };

  class ScheduledEquationsBlocks
  {
    public:
      using iterator = typename impl::ScheduledEquationsBlocks::iterator;
      using const_iterator = typename impl::ScheduledEquationsBlocks::const_iterator;

      ScheduledEquationsBlocks() : impl(std::make_shared<impl::ScheduledEquationsBlocks>())
      {
      }

      /// @name Forwarded methods
      /// {

      std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index)
      {
        return (*impl)[index];
      }

      const std::unique_ptr<ScheduledEquationsBlock>& operator[](size_t index) const
      {
        return (*impl)[index];
      }

      size_t size() const
      {
        return impl->size();
      }

      void append(std::unique_ptr<ScheduledEquationsBlock> block)
      {
        impl->append(std::move(block));
      }

      iterator begin()
      {
        return impl->begin();
      }

      const_iterator begin() const
      {
        return impl->begin();
      }

      iterator end()
      {
        return impl->end();
      }

      const_iterator end() const
      {
        return impl->end();
      }

      /// }

    private:
      std::shared_ptr<impl::ScheduledEquationsBlocks> impl;
  };

  template<>
  class Model<ScheduledEquationsBlock> : public impl::BaseModel
  {
    public:
      Model(mlir::modelica::ModelOp modelOp)
          : impl::BaseModel(std::move(modelOp))
      {
      }

      ScheduledEquationsBlocks getScheduledBlocks() const
      {
        return scheduledBlocks;
      }

      void setScheduledBlocks(ScheduledEquationsBlocks blocks) {
        scheduledBlocks = std::move(blocks);
      }

    private:
      ScheduledEquationsBlocks scheduledBlocks;
  };

  /// Schedule the equations.
  mlir::LogicalResult schedule(Model<ScheduledEquationsBlock>& result, const Model<MatchedEquation>& model);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCHEDULING_H
