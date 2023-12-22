#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_OPTIONS_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_OPTIONS_H

#ifdef SUNDIALS_ENABLE

#include "kinsol/kinsol.h"

namespace marco::runtime::sundials::kinsol
{
  struct Options
  {
    bool debug = false;

    // Relative tolerance is intended as the difference between the values
    // computed through the n-th and the (n+1)-th order BDF method, divided by
    // the absolute value given by the (n+1)-th order BDF method.
    //
    // It is mandatory to set the parameter higher than the minimum precision
    // of the floating point unit roundoff (10^-15 for doubles).
    //
    // It is also highly suggested setting the parameter lower than 10^-3 in
    // order to avoid inaccurate results. KINSOL defaults to 10^-6.
    realtype relativeTolerance = 1e-06;

    // Absolute tolerance is intended as the maximum acceptable difference
    // between the values computed through the n-th and  the (n+1)-th order BDF
    // method.
    //
    // Absolute tolerance is used to substitute relative tolerance when the
    // value converges to zero. When this happens, in fact, the relative error
    // would tend to infinity, thus exceeding the set tolerance.
    //
    // It is mandatory to set the parameter higher than the minimum precision
    // of the floating point unit roundoff (10^-15 for doubles).
    //
    // It is also highly suggested setting the parameter lower than 10^-3 in
    // order to avoid inaccurate results. KINSOL defaults to 10^-6.
    realtype absoluteTolerance = 1e-06;

    // Arbitrary initial guess made in the 20/12/2021 Modelica Call
    realtype maxAlgebraicAbsoluteTolerance = 1e-06;

    realtype fnormtol = 0.000001;
    realtype scsteptol = 0.000001;

    // Whether to print the Jacobian matrices while debugging.
    bool printJacobian = false;

    // Arbitrary initial guesses on 20/12/2021 Modelica Call.
    realtype algebraicTolerance = 1e-12;
    realtype timeScalingFactorInit = 1e5;

    // Maximum number of steps to reach the next output time.
    long maxSteps = 1e4;

    // Initial step size
    realtype initialStepSize = 0;

    // Minimum absolute value of the step size.
    realtype minStepSize = 0;

    // Maximum absolute value of the step size.
    realtype maxStepSize = 0;

    // Maximum number of error test failures in attempting one step.
    int maxErrTestFails = 10;

    // Whether to suppress algebraic variables in the local error test.
    booleantype suppressAlg = SUNFALSE;

    // Maximum number of nonlinear solver iterations in one solve attempt.
    int maxNonlinIters = 4;

    // Maximum number of nonlinear solver convergence failures in one step.
    int maxConvFails = 10;

    // Safety factor in the nonlinear convergence test.
    realtype nonlinConvCoef = 0.33;

    // Positive constant in the Newton iteration convergence test within
    // the initial condition calculation.
    realtype nonlinConvCoefIC = 0.0033;

    // Maximum number of steps allowed for IC
    long maxStepsIC = 5;

    // Maximum number of the approximate Jacobian or preconditioner evaluations
    // allowed when the Newton iteration appears to be slowly converging.
    int maxNumJacsIC = 4;

    // Maximum number of Newton iterations allowed in any one attempt to solve
    // the initial conditions calculation problem.
    int maxNumItersIC = 10;

    // Whether to turn on or off the linesearch algorithm.
    booleantype lineSearchOff = SUNFALSE;

    // The factor multiplying the threads count when computing the total number
    // of equations chunks.
    // In other words, it is the amount of chunks each thread would process in
    // a perfectly balanced scenario.
    int64_t equationsChunksFactor = 10;
  };

  Options& getOptions();
}

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_OPTIONS_H
