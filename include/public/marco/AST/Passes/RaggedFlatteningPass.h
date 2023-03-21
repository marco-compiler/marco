#pragma once

#include "marco/AST/AST.h"
#include "marco/AST/Pass.h"
#include "marco/AST/Symbol.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::ast
{

  class RaggedFlattener : public Pass
  {
    public:
    struct Shape {

      struct DimensionSize {
        using Ragged = llvm::SmallVector<DimensionSize, 3>;
        DimensionSize(long val) : value(val) {}
        DimensionSize(Ragged& val) : value(std::make_unique<Ragged>(std::move(val))) {}
        DimensionSize(const DimensionSize& other)
        {
          if (other.isRagged())
            value = std::make_unique<Ragged>(other.asRagged());
          else
            value = other.asNum();
        }
        DimensionSize& operator=(const DimensionSize& other)
        {
          if (other.isRagged())
            value = std::make_unique<Ragged>(other.asRagged());
          else
            value = other.asNum();
          return *this;
        }

        bool isRagged() const
        {
          return !std::holds_alternative<long>(value);
        }
        const Ragged& asRagged() const
        {
          return *std::get<std::unique_ptr<Ragged>>(value);
        }
        long asNum() const
        {
          return std::get<long>(value);
        }
        std::variant<long, std::unique_ptr<Ragged>> value;
      };
      llvm::ArrayRef<DimensionSize> dimensions() const
      {
        return dims;
      }

      Shape(llvm::SmallVector<DimensionSize, 3>& dims) : dims(std::move(dims)) {}
      Shape() {}
      llvm::SmallVector<DimensionSize, 3> dims;
    };

    using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
    using ShapeTable = llvm::ScopedHashTable<llvm::StringRef, Shape>;
    using TranslationTable = llvm::ScopedHashTable<llvm::StringRef, int>;

    RaggedFlattener(diagnostic::DiagnosticEngine& diagnostics);

    bool run(std::unique_ptr<Class>& cls) override;

    bool run(Algorithm& algorithm);

    template<typename T>
    bool run(Class& cls);

    bool run(Equation& equation);

    template<typename T>
    bool run(Expression& expression);

    bool run(ForEquation& forEquation);
    bool run(Induction& induction);
    bool run(Member& member);

    template<typename T>
    bool run(Statement& statement);

    private:
    Model* getModel();

    SymbolTable symbolTable;
    ShapeTable shapeTable;
    TranslationTable translationTable;

    bool removeFlag;
    llvm::SmallVector<ASTNode*, 8> parentStack;
    llvm::SmallVector<std::unique_ptr<ForEquation>, 2> forEquationsToAdd;
    llvm::SmallVector<std::unique_ptr<Equation>, 2> equationsToAdd;
    llvm::Optional<int> range_index;
  };

  template<>
  bool RaggedFlattener::run<Class>(Class& cls);

  template<>
  bool RaggedFlattener::run<PartialDerFunction>(Class& cls);

  template<>
  bool RaggedFlattener::run<StandardFunction>(Class& cls);

  template<>
  bool RaggedFlattener::run<Model>(Class& cls);

  template<>
  bool RaggedFlattener::run<Package>(Class& cls);

  template<>
  bool RaggedFlattener::run<Record>(Class& cls);

  template<>
  bool RaggedFlattener::run<Expression>(Expression& expression);

  template<>
  bool RaggedFlattener::run<Array>(Expression& expression);

  template<>
  bool RaggedFlattener::run<Call>(Expression& expression);

  template<>
  bool RaggedFlattener::run<Constant>(Expression& expression);

  template<>
  bool RaggedFlattener::run<Operation>(Expression& expression);

  template<>
  bool RaggedFlattener::run<ReferenceAccess>(Expression& expression);

  template<>
  bool RaggedFlattener::run<Tuple>(Expression& expression);

  template<>
  bool RaggedFlattener::run<AssignmentStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<BreakStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<ForStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<IfStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<ReturnStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<WhenStatement>(Statement& statement);

  template<>
  bool RaggedFlattener::run<WhileStatement>(Statement& statement);

  std::unique_ptr<Pass> createRaggedFlatteningPass(diagnostic::DiagnosticEngine& diagnostics);
}// namespace marco::ast
