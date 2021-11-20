#include "TestCases.h"

namespace marco::matching
{
  MatchingGraph<Variable, Equation> testCase1()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 3 });
    graph.addVariable(x);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 2));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 1)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(2)));
    graph.addEquation(eq2);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase2()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 4 });
    graph.addVariable(x);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 2));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(2, 4));
    eq2.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq2);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase3()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable l("l", { 1 });
    graph.addVariable(l);

    Variable h("h", { 1 });
    graph.addVariable(h);

    Variable fl("fl", { 1 });
    graph.addVariable(fl);

    Variable fh("fh", { 1 });
    graph.addVariable(fh);

    Variable x("x", { 5 });
    graph.addVariable(x);

    Variable y("y", { 5 });
    graph.addVariable(y);

    Variable f("f", { 5 });
    graph.addVariable(f);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 1));
    eq1.addVariableAccess(Access(l, DimensionAccess::constant(0)));
    eq1.addVariableAccess(Access(fl, DimensionAccess::constant(0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(fl, DimensionAccess::constant(0)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 1));
    eq3.addVariableAccess(Access(h, DimensionAccess::constant(0)));
    eq3.addVariableAccess(Access(fh, DimensionAccess::constant(0)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(0, 1));
    eq4.addVariableAccess(Access(fh, DimensionAccess::constant(0)));
    graph.addEquation(eq4);

    Equation eq5("eq5");
    eq5.addIterationRange(Range(0, 5));
    eq5.addVariableAccess(Access(fl, DimensionAccess::constant(0)));
    eq5.addVariableAccess(Access(f, DimensionAccess::relative(0, 0)));
    eq5.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq5);

    Equation eq6("eq6");
    eq6.addIterationRange(Range(0, 5));
    eq6.addVariableAccess(Access(fh, DimensionAccess::constant(0)));
    eq6.addVariableAccess(Access(f, DimensionAccess::relative(0, 0)));
    eq6.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq6);

    Equation eq7("eq7");
    eq7.addIterationRange(Range(0, 5));
    eq7.addVariableAccess(Access(f, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq7);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase4()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable l("l", { 1 });
    graph.addVariable(l);

    Variable h("h", { 1 });
    graph.addVariable(h);

    Variable x("x", { 5 });
    graph.addVariable(x);

    Variable f("f", { 6 });
    graph.addVariable(f);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 1));
    eq1.addVariableAccess(Access(l, DimensionAccess::constant(0)));
    eq1.addVariableAccess(Access(f, DimensionAccess::constant(0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(f, DimensionAccess::constant(0)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 5));
    eq3.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq3.addVariableAccess(Access(f, DimensionAccess::relative(0, 0)));
    eq3.addVariableAccess(Access(f, DimensionAccess::relative(0, 1)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(1, 5));
    eq4.addVariableAccess(Access(f, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq4);

    Equation eq5("eq5");
    eq5.addIterationRange(Range(0, 1));
    eq5.addVariableAccess(Access(h, DimensionAccess::constant(0)));
    eq5.addVariableAccess(Access(f, DimensionAccess::constant(5)));
    graph.addEquation(eq5);

    Equation eq6("eq6");
    eq6.addIterationRange(Range(0, 1));
    eq6.addVariableAccess(Access(f, DimensionAccess::constant(5)));
    graph.addEquation(eq6);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase5()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 5 });
    graph.addVariable(x);

    Variable y("y", { 4 });
    graph.addVariable(y);

    Variable z("z", { 5 });
    graph.addVariable(z);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 5));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 4));
    eq2.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    eq2.addVariableAccess(Access(x, DimensionAccess::relative(0, 1)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 4));
    eq3.addVariableAccess(Access(z, DimensionAccess::relative(0, 0)));
    eq3.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq3.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(0, 1));
    eq4.addVariableAccess(Access(z, DimensionAccess::constant(4)));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(4)));
    graph.addEquation(eq4);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase6()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 6 });
    graph.addVariable(x);

    Variable y("y", { 3 });
    graph.addVariable(y);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 3));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq1.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 6));
    eq2.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq2.addVariableAccess(Access(y, DimensionAccess::constant(1)));
    graph.addEquation(eq2);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase7()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 2 });
    graph.addVariable(x);

    Variable y("y", { 1 });
    graph.addVariable(y);

    Variable z("z", { 1 });
    graph.addVariable(z);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 1));
    eq1.addVariableAccess(Access(x, DimensionAccess::constant(0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(1)));
    eq2.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 1));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    eq3.addVariableAccess(Access(z, DimensionAccess::constant(0)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(0, 1));
    eq4.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    eq4.addVariableAccess(Access(z, DimensionAccess::constant(0)));
    graph.addEquation(eq4);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase8()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 9 });
    graph.addVariable(x);

    Variable y("y", { 3 });
    graph.addVariable(y);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 3));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq1.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(3, 7));
    eq2.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq2.addVariableAccess(Access(y, DimensionAccess::constant(1)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(7, 9));
    eq3.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(2)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(0, 3));
    eq4.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq4);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase9()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 5 });
    graph.addVariable(x);

    Variable y("y", { 5 });
    graph.addVariable(y);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 5));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq1.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(0)));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(1)));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(2)));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(3)));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(4)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 1));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(1)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(2)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(3)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(4)));
    graph.addEquation(eq3);

    Equation eq4("eq4");
    eq4.addIterationRange(Range(0, 1));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(0)));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(1)));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(2)));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(3)));
    eq4.addVariableAccess(Access(x, DimensionAccess::constant(4)));
    graph.addEquation(eq4);

    Equation eq5("eq5");
    eq5.addIterationRange(Range(0, 1));
    eq5.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    eq5.addVariableAccess(Access(y, DimensionAccess::constant(1)));
    eq5.addVariableAccess(Access(y, DimensionAccess::constant(2)));
    eq5.addVariableAccess(Access(y, DimensionAccess::constant(3)));
    eq5.addVariableAccess(Access(y, DimensionAccess::constant(4)));
    graph.addEquation(eq5);

    Equation eq6("eq6");
    eq6.addIterationRange(Range(0, 1));
    eq6.addVariableAccess(Access(x, DimensionAccess::constant(0)));
    eq6.addVariableAccess(Access(x, DimensionAccess::constant(1)));
    eq6.addVariableAccess(Access(x, DimensionAccess::constant(2)));
    eq6.addVariableAccess(Access(x, DimensionAccess::constant(3)));
    eq6.addVariableAccess(Access(x, DimensionAccess::constant(4)));
    graph.addEquation(eq6);

    return graph;
  }

  MatchingGraph<Variable, Equation> testCase10()
  {
    MatchingGraph<Variable, Equation> graph;

    Variable x("x", { 2 });
    graph.addVariable(x);

    Variable y("y", { 2 });
    graph.addVariable(y);

    Equation eq1("eq1");
    eq1.addIterationRange(Range(0, 2));
    eq1.addVariableAccess(Access(x, DimensionAccess::relative(0, 0)));
    eq1.addVariableAccess(Access(y, DimensionAccess::relative(0, 0)));
    graph.addEquation(eq1);

    Equation eq2("eq2");
    eq2.addIterationRange(Range(0, 1));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(0)));
    eq2.addVariableAccess(Access(x, DimensionAccess::constant(1)));
    graph.addEquation(eq2);

    Equation eq3("eq3");
    eq3.addIterationRange(Range(0, 1));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(0)));
    eq3.addVariableAccess(Access(y, DimensionAccess::constant(1)));
    graph.addEquation(eq3);

    return graph;
  }
}

