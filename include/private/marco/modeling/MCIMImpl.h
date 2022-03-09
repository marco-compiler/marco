#ifndef MARCO_MODELING_MCIMIMPL_H
#define MARCO_MODELING_MCIMIMPL_H

#include "llvm/Support/Casting.h"
#include "marco/modeling/MCIM.h"

namespace marco::modeling::internal
{
  class MCIM::Impl
  {
    public:
      enum MCIMKind
      {
        Regular,
        Flat
      };

      Impl(MCIMKind kind, MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

      Impl(const Impl& other);

      virtual ~Impl();

      /// @name LLVM-style RTTI methods
      /// {

      MCIMKind getKind() const
      {
        return kind;
      }

      template<typename T>
      bool isa() const
      {
        return llvm::isa<T>(this);
      }

      template<typename T>
      T* dyn_cast()
      {
        return llvm::dyn_cast<T>(this);
      }

      template<typename T>
      const T* dyn_cast() const
      {
        return llvm::dyn_cast<T>(this);
      }

      /// }
      /// @name Forwarding methods
      /// {

      virtual bool operator==(const MCIM::Impl& rhs) const;

      virtual bool operator!=(const MCIM::Impl& rhs) const;

      virtual std::unique_ptr<MCIM::Impl> clone() = 0;

      const MultidimensionalRange& getEquationRanges() const;

      const MultidimensionalRange& getVariableRanges() const;

      llvm::iterator_range<IndexesIterator> getIndexes() const;

      virtual MCIM::Impl& operator+=(const MCIM::Impl& rhs);

      virtual MCIM::Impl& operator-=(const MCIM::Impl& rhs);

      virtual void apply(const AccessFunction& access) = 0;

      virtual bool get(const Point& equation, const Point& variable) const = 0;

      virtual void set(const Point& equation, const Point& variable) = 0;

      virtual void unset(const Point& equation, const Point& variable) = 0;

      virtual bool empty() const = 0;

      virtual void clear() = 0;

      virtual IndexSet flattenRows() const = 0;

      virtual IndexSet flattenColumns() const = 0;

      virtual std::unique_ptr<MCIM::Impl> filterRows(const IndexSet& filter) const = 0;

      virtual std::unique_ptr<MCIM::Impl> filterColumns(const IndexSet& filter) const = 0;

      virtual std::vector<std::unique_ptr<MCIM::Impl>> splitGroups() const = 0;

      /// }

    private:
      const MCIMKind kind;
      MultidimensionalRange equationRanges;
      MultidimensionalRange variableRanges;
  };
}

#endif // MARCO_MODELING_MCIMIMPL_H
