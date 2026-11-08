#include "marco/Codegen/Lowering/Modelica/Lowerer.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Parser/Modelica/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering::modelica;
using ::testing::HasSubstr;
using ::testing::Not;

namespace {
/// Result of the lowering of Modelica sources.
struct LoweringResult {
  bool parsed = true;
  bool succeeded = false;
  bool verified = false;
  std::string ir;
  std::vector<std::string> errors;
};

/// Parse and lower the given sources, in order.
LoweringResult lower(llvm::ArrayRef<llvm::StringRef> sources) {
  LoweringResult result;

  std::vector<std::unique_ptr<llvm::MemoryBuffer>> buffers;
  std::vector<std::shared_ptr<SourceFile>> files;
  std::vector<std::unique_ptr<ast::modelica::StoredDefinition>> roots;

  for (auto [index, source] : llvm::enumerate(sources)) {
    std::string fileName = "test" + std::to_string(index) + ".mo";
    auto file = std::make_shared<SourceFile>(fileName);
    auto buffer = llvm::MemoryBuffer::getMemBuffer(source);
    file->setMemoryBuffer(buffer.get());

    clang::DiagnosticOptions diagOpts;
    clang::DiagnosticsEngine diagnostics(
        llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
            new clang::DiagnosticIDs()),
        diagOpts, new clang::IgnoringDiagConsumer());

    clang::SourceManagerForFile sourceManager(fileName, source);
    parser::modelica::Parser parser(diagnostics, sourceManager.get(), file);
    auto root = parser.parseStoredDefinition();

    if (!root) {
      result.parsed = false;
      return result;
    }

    roots.push_back(std::move(root));
    files.push_back(file);
    buffers.push_back(std::move(buffer));
  }

  mlir::MLIRContext context;
  context.loadDialect<mlir::modelica::ModelicaDialect>();

  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    result.errors.push_back(diag.str());
    return mlir::success();
  });

  Lowerer lowerer(context);

  for (const auto &root : roots) {
    if (mlir::failed(lowerer.addFile(*root))) {
      return result;
    }
  }

  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  result.succeeded = mlir::succeeded(lowerer.lower(module));
  result.verified = mlir::succeeded(mlir::verify(module));

  llvm::raw_string_ostream os(result.ir);
  module.print(os);
  module->erase();
  return result;
}

LoweringResult lower(llvm::StringRef source) {
  return lower(llvm::ArrayRef<llvm::StringRef>(source));
}

/// Expect the lowering to succeed and produce valid IR.
#define EXPECT_LOWERED(result)                                                 \
  do {                                                                         \
    ASSERT_TRUE((result).parsed);                                              \
    EXPECT_TRUE((result).succeeded) << ::testing::PrintToString((result).errors); \
    EXPECT_TRUE((result).verified) << (result).ir;                             \
  } while (false)
} // namespace

//===---------------------------------------------------------------------===//
// Classes and types
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, classes) {
  auto result = lower(R"(
    package P "A package"
      partial model M end M;
      connector C end C;
      impure function f end f;
      type E = enumeration(a "First", b);
      function df = der(f, x);
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.class package @P attributes "
                                   "{description = \"A package\"}"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.class partial model @M"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.class connector @C"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.class impure function @f"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.enumeration @E"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.enum_literal @a \"First\""));
  EXPECT_THAT(result.ir, HasSubstr("modelica.class function @df der(@f, \"x\")"));
}

TEST(ModelicaLowering, componentTypes) {
  auto result = lower(R"(
    package P
      type Voltage = Real(unit = "V");
      type Vec3 = Real[3];
      type E = enumeration(x, y, z);
      model M
        Voltage v;
        Vec3 a[2];
        Real[4] b[n];
        Real c[Boolean, E];
        E e;
        parameter Integer n = 1;
      end M;
    end P;)");

  EXPECT_LOWERED(result);

  EXPECT_THAT(result.ir, HasSubstr("modelica.component @v : @Voltage -> "
                                   "!modelica.real"));

  // The dimensions of the declaration precede the ones of the type.
  EXPECT_THAT(result.ir, HasSubstr("modelica.component @a : @Vec3 -> "
                                   "tensor<2x3x!modelica.real> dims [2]"));

  EXPECT_THAT(result.ir, HasSubstr("modelica.component @b : @Real -> "
                                   "tensor<?x4x!modelica.real> dims [?, 4]"));

  EXPECT_THAT(result.ir,
              HasSubstr("modelica.component @c : @Real -> "
                        "tensor<2x3x!modelica.real> dims [Boolean, @E]"));

  EXPECT_THAT(result.ir, HasSubstr("modelica.component @e : @E -> "
                                   "!modelica.enum<@P::@E>"));

  EXPECT_THAT(result.ir, HasSubstr("modelica.component parameter @n : "
                                   "@Integer -> !modelica.int"));
}

TEST(ModelicaLowering, shortClassesAndExtends) {
  auto result = lower(R"(
    package P
      connector RealInput = input Real "Input";
      type Vec = Real[3](each min = 0);
      model Base end Base;
      model D
        extends Base;
        extends .P.Base annotation(IconMap(primitivesVisible = false));
      end D;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.class short connector @RealInput"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.extends @Real input"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.modify each @min"));
  EXPECT_THAT(result.ir, HasSubstr("dims [3]"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.extends @Base"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.extends @\".\"::@P::@Base "
                                   "attributes {annotation = {IconMap = "
                                   "{primitivesVisible = false}}}"));
}

TEST(ModelicaLowering, withinClauses) {
  auto result = lower({"package P end P;", "within P; model M end M;"});

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.class package @P {\n"
                                   "    modelica.class model @M"));
}

//===---------------------------------------------------------------------===//
// Name lookup
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, inheritedElements) {
  auto result = lower(R"(
    package P
      connector Pin
        Real v;
        flow Real i;
      end Pin;
      partial model TwoPin
        Pin p, n;
        Real v;
      end TwoPin;
      model Resistor
        extends TwoPin;
        parameter Real R = 1;
      equation
        v = R * p.i;
      end Resistor;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.ref @v : !modelica.real"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.ref @p : !modelica.object<@P::@Pin>"));
  EXPECT_THAT(result.ir, HasSubstr("@i : (!modelica.object<@P::@Pin>) -> "
                                   "!modelica.real"));
}

TEST(ModelicaLowering, redeclarationsOfInheritedClasses) {
  // The inherited input has the type redeclared by the derived package.
  auto result = lower(R"(
    package P
      partial package Base
        replaceable record State end State;
        replaceable partial function f
          input State s;
          output Real y;
        end f;
      end Base;
      package Impl
        extends Base;
        redeclare record extends State
          Real p;
        end State;
        redeclare function extends f
        algorithm
          y := s.p;
        end f;
      end Impl;
      model M
        Real y = Impl.f(Impl.State(p = 1));
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.member %1, @p : "
                                   "(!modelica.object<@P::@Impl::@State>) -> "
                                   "!modelica.real"));

  // The output of the class-extends function is inherited.
  EXPECT_THAT(result.ir,
              HasSubstr("modelica.call @Impl::@f(%1) : "
                        "(!modelica.object<@P::@Impl::@State>) -> "
                        "!modelica.real"));
}

TEST(ModelicaLowering, redeclarationsInExtendsClauses) {
  auto result = lower(R"(
    package P
      record A Real x; end A;
      record B extends A; Real y; end B;
      partial model Base
        replaceable A a;
      end Base;
      model M
        extends Base(redeclare B a);
        Real z = a.y;
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.ref @a : !modelica.object<@P::@B>"));
}

TEST(ModelicaLowering, imports) {
  auto result = lower(R"(
    package Lib
      package Constants
        constant Real pi = 3.14;
      end Constants;
    end Lib;
    package P
      import C = Lib.Constants;
      import Lib.Constants.*;
      model M
        Real a = C.pi;
        Real b = pi;
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.import @C = @Lib::@Constants"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.import @Lib::@Constants unqualified"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.ref @C::@pi : !modelica.real"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.ref @pi : !modelica.real"));
}

TEST(ModelicaLowering, encapsulatedClasses) {
  // Lookup stops at encapsulated classes, except for predefined types.
  auto result = lower(R"(
    package P
      constant Real k = 1;
      encapsulated model M
        Real x = k;
      end M;
    end P;)");

  EXPECT_FALSE(result.succeeded);
  EXPECT_THAT(result.errors, ::testing::Contains("unknown name 'k'"));
}

TEST(ModelicaLowering, enumerationLiterals) {
  auto result = lower(R"(
    package P
      type E = enumeration(a, b, c);
      model M
        E e = E.c;
        StateSelect s = StateSelect.prefer;
        Real x[E];
      equation
        for i in E loop
          x[i] = 0;
        end for;
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("#modelica<enum \"c\" = 3> : "
                                   "!modelica.enum<@P::@E>"));
  EXPECT_THAT(result.ir, HasSubstr("#modelica<enum \"prefer\" = 4> : "
                                   "!modelica.enum<@StateSelect>"));

  // Enumeration types used as iteration ranges.
  EXPECT_THAT(result.ir, HasSubstr("#modelica<array [\"a\" = 1, \"b\" = 2, "
                                   "\"c\" = 3]>"));
}

//===---------------------------------------------------------------------===//
// Expressions
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, operatorsAndLiterals) {
  auto result = lower(R"(
    model M
      Real x = -2 * (3 + 4.5) ^ 2 / 7'm';
      Boolean b = not (x > 1 and x <= 2);
      Real r[:] = 1:2:10;
      Real y = if b then 1 elseif x > 0 then 2 else 3.5;
    end M;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("#modelica<int 7 unit \"m\">"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.pow"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.neg"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.range %0 to %1 step %2"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.if_expr"));
}

TEST(ModelicaLowering, ifExpressionTypes) {
  // The result is Real, and arrays of different sizes lead to dynamic sizes.
  auto result = lower(R"(
    model M
      parameter Boolean b = true;
      Real y = if b then 1 else 2.5;
      Real z[:] = if b then {1, 2} else {1, 2, 3};
    end M;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.if_expr %0 : !modelica.real"));
  EXPECT_THAT(result.ir, HasSubstr(": tensor<?x!modelica.int>"));
}

TEST(ModelicaLowering, subscriptsAndEnd) {
  auto result = lower(R"(
    model M
      Real a[3, 4];
      Real b[4] = a[end, :];
      Real c[2] = a[{1, 2}, 1];
    end M;)");

  // Indices are zero-based in the dialect: 'end' is the size of the
  // dimension minus one, and literal indices are shifted directly.
  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("%1 = modelica.constant #modelica<int 0>"));
  EXPECT_THAT(result.ir, HasSubstr("%2 = modelica.size %0, %1 : "
                                   "(tensor<3x4x!modelica.real>, !modelica.int) "
                                   "-> !modelica.int"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.sub %2, %3 : "
                                   "(!modelica.int, !modelica.int) "
                                   "-> !modelica.int"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.subscript %0[%4, :] : "
                                   "(tensor<3x4x!modelica.real>, !modelica.int) "
                                   "-> tensor<4x!modelica.real>"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.constant #modelica<int 1>"));
  EXPECT_THAT(result.ir, Not(HasSubstr("modelica.sub_ew")));
  EXPECT_THAT(result.ir, HasSubstr("-> tensor<2x!modelica.real>"));
}

TEST(ModelicaLowering, arrayConstructors) {
  auto result = lower(R"(
    model M
      Real a[2, 3] = {{1, 2, 3}, {4, 5, 6}};
      Real b[2, 3] = [1, 2, 3; 4, 5, 6];
      Real c[3, 2] = {i * j for i in 1:2, j in 1:3};
    end M;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("-> tensor<2x3x!modelica.int>"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.cat"));

  // The last iterator varies along the outermost dimension.
  EXPECT_THAT(result.ir, HasSubstr("-> tensor<3x2x!modelica.int>"));
}

TEST(ModelicaLowering, builtins) {
  auto result = lower(R"(
    model M
      Real x[3];
      Real s = sum(x[i] for i in 1:3) + sum(x) + min(x) + max(1, 2);
      Integer n = size(x, 1) + integer(2.5) + div(7, 2);
      Real z[2, 2] = zeros(2, 2) + identity(2);
      String str = String(s, significantDigits = 3);
    equation
      der(x) = -x;
    end M;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.reduction sum"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.array_reduce sum"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.array_reduce min"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.max"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.size"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.integer"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.div_trunc"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.zeros"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.identity"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.to_string %0 significantDigits"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.der"));
}

TEST(ModelicaLowering, userFunctionsShadowBuiltins) {
  auto result = lower(R"(
    package P
      function sum
        input Real x;
        output Real y = x;
      end sum;
      model M
        Real y = sum(2);
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.call @sum("));
}

TEST(ModelicaLowering, functionCalls) {
  auto result = lower(R"(
    package P
      function f
        input Real x;
        output Real a;
        output Integer b;
      algorithm
        a := x;
        b := 1;
      end f;
      function g
        input Real x;
        output Real y = 2 * x;
      end g;
      record R Real x; end R;
      model M
        Real a, c[3], v[3];
        Integer b;
        R r = R(x = 1);
      equation
        (a, b) = f(1);
        (a, ) = f(2);
        c = g(v);
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.call @f(%2) : (!modelica.int) -> "
                                   "(!modelica.real, !modelica.int)"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.equation_sides %0, %1 = %3#0, %3#1"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.equation_sides %0 = %2#0"));

  // Record constructor.
  EXPECT_THAT(result.ir, HasSubstr("modelica.call @R(@x = %0) : "
                                   "(!modelica.int) -> !modelica.object<@P::@R>"));

  // Automatic vectorization.
  EXPECT_THAT(result.ir, HasSubstr("modelica.call @g(%1) : "
                                   "(tensor<3x!modelica.real>) -> "
                                   "tensor<3x!modelica.real>"));
}

TEST(ModelicaLowering, externalObjects) {
  auto result = lower(R"(
    package P
      class Table
        extends ExternalObject;
        function constructor
          input Real x;
          output Table table;
        external "C" table = init(x);
        end constructor;
      end Table;
      model M
        Table t = Table(1);
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.call @Table(%0) : (!modelica.int) "
                                   "-> !modelica.object<@P::@Table>"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.external \"C\" \"init\" output"));
}

//===---------------------------------------------------------------------===//
// Modifications
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, modifications) {
  auto result = lower(R"(
    package P
      record A Real x; Real y; end A;
      record B extends A; end B;
      model M
        A a[2](each final x = 1, y(start = 2) = {3, 4});
        Real z(start = 1) = break;
        parameter Integer n = 2;
        Real w[n] if n > 1;
      end M;
      model N
        M m(redeclare B a, n = 3);
      end N;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.modify each final @x"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.modify @y {\n"
                                   "          modelica.modify @start"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.bind break"));
  EXPECT_THAT(result.ir, HasSubstr("dims [?] {"));
  EXPECT_THAT(result.ir, HasSubstr("if {"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.component redeclare @a : @B -> "
                                   "!modelica.object<@P::@B>"));
}

//===---------------------------------------------------------------------===//
// Equations and statements
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, equations) {
  auto result = lower(R"(
    package P
      connector Pin Real v; flow Real i; end Pin;
      model M
        parameter Integer n = 3;
        Pin p[n];
        Real x;
        discrete Real d;
      equation
        for i in 1:n - 1 loop
          connect(p[i], p[i + 1]);
        end for;
        if n > 2 then
          x = 1;
        elseif n > 1 then
          x = 2;
        else
          x = 3;
        end if;
        when sample(0, 1) then
          d = pre(d) + 1;
        elsewhen initial() then
          reinit(x, 0);
        end when;
        assert(x > 0, "message", AssertionLevel.warning);
      initial equation
        d = 0;
      end M;
    end P;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.for_equation %arg0 : !modelica.int in"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.connect {"));
  EXPECT_THAT(result.ir, HasSubstr("} else {\n          modelica.if_equation"));
  EXPECT_THAT(result.ir, HasSubstr("} elsewhen {\n          modelica.when_equation"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.reinit {"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.assert {"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.equations initial {"));
}

TEST(ModelicaLowering, statements) {
  auto result = lower(R"(
    function f
      input Real x[:];
      output Real y;
      output Integer k;
    protected
      Integer i;
    algorithm
      y := 0;
      for j in 1:size(x, 1) loop
        if x[j] < 0 then
          break;
        end if;
        y := y + x[j];
      end for;
      while y > 10 loop
        y := y / 2;
      end while;
      (y, k) := g(y);
      return;
    end f;
    function g
      input Real x;
      output Real y = x;
      output Integer k = 1;
    end g;)");

  EXPECT_LOWERED(result);
  EXPECT_THAT(result.ir, HasSubstr("modelica.for %arg0 : !modelica.int in"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.break"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.while {"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.assignment_sides %0, %1 = %3#0, %3#1"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.return"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.component protected @i"));
}

//===---------------------------------------------------------------------===//
// Errors
//===---------------------------------------------------------------------===//

TEST(ModelicaLowering, failingElementsAreOmitted) {
  auto result = lower(R"(
    package P
      model M
        Real a = unknown;
        Real b = 1;
        model Nested end Nested;
      end M;
    end P;)");

  EXPECT_FALSE(result.succeeded);
  EXPECT_TRUE(result.verified);
  EXPECT_THAT(result.errors, ::testing::Contains("unknown name 'unknown'"));

  // The other elements and the nested classes are kept.
  EXPECT_THAT(result.ir, Not(HasSubstr("modelica.component @a")));
  EXPECT_THAT(result.ir, HasSubstr("modelica.component @b"));
  EXPECT_THAT(result.ir, HasSubstr("modelica.class model @Nested"));
}

TEST(ModelicaLowering, operatorOverloadingIsReported) {
  auto result = lower(R"(
    package P
      operator record Complex Real re, im; end Complex;
      model M
        Complex a, b;
        Complex c = a + b;
      end M;
    end P;)");

  EXPECT_FALSE(result.succeeded);
  EXPECT_TRUE(result.verified);
  EXPECT_THAT(result.errors,
              ::testing::Contains(HasSubstr("operands must be numeric")));
}

TEST(ModelicaLowering, missingWithinPackage) {
  auto result = lower("within Missing; model M end M;");
  EXPECT_FALSE(result.succeeded);
  EXPECT_THAT(result.errors,
              ::testing::Contains("the package 'Missing' of the within-clause "
                                  "is not defined"));
}
