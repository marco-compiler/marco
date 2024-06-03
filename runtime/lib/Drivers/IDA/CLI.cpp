#ifdef CLI_ENABLE
#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include <iostream>

namespace marco::runtime::sundials::ida
{
  std::string CommandLineOptions::getTitle() const
  {
    return "IDA";
  }

  void CommandLineOptions::printCommandLineOptions(
      std::ostream& os) const
  {
    os << "  --time-step=<value>                    Set the time step (in seconds)." << std::endl;

    os << "  --ida-equations-chunks-factor          Set the factor which, once multiplied by the threads count, determines the number of equation chunks. Defaults to " << getOptions().equationsChunksFactor << "." << std::endl;

    os << "  --ida-relative-tolerance=<value>       Set the relative tolerance. Defaults to " << getOptions().relativeTolerance << "." << std::endl;
    os << "  --ida-absolute-tolerance=<value>       Set the absolute tolerance. Defaults to " << getOptions().absoluteTolerance << "." << std::endl;
    os << "  --ida-max-algebraic-abs-tol=<value>    Set the maximum absolute tolerance allowed for algebraic variables. Defaults to " << getOptions().maxAlgebraicAbsoluteTolerance << "." << std::endl;
    os << "  --ida-time-scaling-factor-ic=<value>   Set the dividing factor for the initial step size guess (the higher the value, the smaller the step). Defaults to " << getOptions().timeScalingFactorInit << "." << std::endl;

    os << "  --ida-max-steps=<value>                Set the maximum number of steps to be taken by the solver in its attempt to reach the next output time. Defaults to " << getOptions().maxSteps << "." << std::endl;
    os << "  --ida-initial-step-size=<value>        Set the initial step size. Defaults to " << getOptions().initialStepSize << "." << std::endl;
    os << "  --ida-min-step-size=<value>            Set the minimum absolute value of the step size. Defaults to " << getOptions().minStepSize << "." << std::endl;
    os << "  --ida-max-step-size=<value>            Set the maximum absolute value of the step size. Defaults to " << getOptions().maxStepSize << "." << std::endl;
    os << "  --ida-max-err-test-fails=<value>       Set the maximum number of error test failures in attempting one step. Defaults to " << getOptions().maxErrTestFails << "." << std::endl;
    os << "  --ida-suppress-alg-vars                Suppress algebraic variables in the local error test." << std::endl;
    os << "  --ida-max-nonlin-iters=<value>         Maximum number of nonlinear solver iterations in one solve attempt. Defaults to " << getOptions().maxNonlinIters << "." << std::endl;
    os << "  --ida-max-conv-fails=<value>           Maximum number of nonlinear solver convergence failures in one step. Defaults to " << getOptions().maxConvFails << "." << std::endl;
    os << "  --ida-nonlin-conv-coef=<value>         Safety factor in the nonlinear convergence test. Defaults to " << getOptions().nonlinConvCoef << "." << std::endl;
    os << "  --ida-nonlin-conv-coef-ic=<value>      Positive constant in the Newton iteration convergence test within the initial condition calculation. Defaults to " << getOptions().nonlinConvCoefIC << "." << std::endl;
    os << "  --ida-max-steps-ic=<value>             Maximum number of steps allowed for IC. Defaults to " << getOptions().maxStepsIC << "." << std::endl;
    os << "  --ida-max-jacs-ic=<value>              Maximum number of the approximate Jacobian or preconditioner evaluations allowed when the Newton iteration appears to be slowly converging. Defaults to " << getOptions().maxNumJacsIC << "." << std::endl;
    os << "  --ida-max-iters-ic=<value>             Maximum number of Newton iterations allowed in any one attempt to solve the initial conditions calculation problem. Defaults to " << getOptions().maxNumItersIC << "." << std::endl;
    os << "  --ida-line-search-off                  Disable the linesearch algorithm." << std::endl;

    os << "  --ida-print-jacobian                   Whether to print the Jacobian matrices while debugging." << std::endl;
  }

  void CommandLineOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    getOptions().equidistantTimeGrid = static_cast<bool>(
        options("time-step") >> getOptions().timeStep);

    options("ida-equations-chunks-factor") >> getOptions().equationsChunksFactor;

    options("ida-max-steps") >> getOptions().maxSteps;
    options("ida-initial-step-size") >> getOptions().initialStepSize;
    options("ida-min-step-size") >> getOptions().minStepSize;
    options("ida-max-step-size") >> getOptions().maxStepSize;
    options("ida-max-err-test-fails") >> getOptions().maxErrTestFails;
    getOptions().suppressAlg = options["ida-suppress-alg-vars"] ? SUNTRUE : SUNFALSE;
    options("ida-max-nonlin-iters") >> getOptions().maxNonlinIters;
    options("ida-max-conv-fails") >> getOptions().maxConvFails;
    options("ida-nonlin-conv-coef") >> getOptions().nonlinConvCoef;
    options("ida-nonlin-conv-coef-ic") >> getOptions().nonlinConvCoefIC;
    options("ida-max-steps-ic") >> getOptions().maxStepsIC;
    options("ida-max-jacs-ic") >> getOptions().maxNumJacsIC;
    options("ida-max-iters-ic") >> getOptions().maxNumItersIC;
    getOptions().lineSearchOff = options["ida-line-search-off"] ? SUNTRUE : SUNFALSE;

    getOptions().printJacobian = options["ida-print-jacobian"];
  }
}

#endif // SUNDIALS_ENABLE
#endif // CLI_ENABLE
