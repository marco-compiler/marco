#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"

namespace marco::runtime::ida
{
  std::string CommandLineOptions::getTitle() const
  {
    return "IDA";
  }

  void CommandLineOptions::printCommandLineOptions(std::ostream& os) const
  {
    os << "  --ida-relative-tolerance=<value>     Set the relative tolerance" << std::endl;
    os << "  --ida-absolute-tolerance=<value>     Set the absolute tolerance" << std::endl;
    os << "  --ida-max-algebraic-abs-tol=<value>  Set the maximum absolute tolerance allowed for algebraic variables" << std::endl;
    os << "  --ida-time-scaling-factor-ic=<value> Set the dividing factor for the initial step size guess (the higher the value, the smaller the step)" << std::endl;

    os << "  --ida-max-steps=<value>              Set the maximum number of steps to be taken by the solver in its attempt to reach the next output time" << std::endl;
    os << "  --ida-initial-step-size=<value>      Set the initial step size" << std::endl;
    os << "  --ida-min-step-size=<value>          Set the minimum absolute value of the step size" << std::endl;
    os << "  --ida-max-step-size=<value>          Set the maximum absolute value of the step size" << std::endl;
    os << "  --ida-max-err-test-fails=<value>     Set the maximum number of error test failures in attempting one step" << std::endl;
    os << "  --ida-suppress-alg-vars              Suppress algebraic variables in the local error test" << std::endl;
    os << "  --ida-max-nonlin-iters=<value>       Maximum number of nonlinear solver iterations in one solve attempt" << std::endl;
    os << "  --ida-max-conv-fails=<value>         Maximum number of nonlinear solver convergence failures in one step" << std::endl;
    os << "  --ida-nonlin-conv-coef=<value>       Safety factor in the nonlinear convergence test" << std::endl;
    os << "  --ida-nonlin-conv-coef-ic=<value>    Positive constant in the Newton iteration convergence test within the initial condition calculation" << std::endl;
    os << "  --ida-max-steps-ic=<value>           Maximum number of steps allowed for IC" << std::endl;
    os << "  --ida-max-jacs-ic=<value>            Maximum number of the approximate Jacobian or preconditioner evaluations allowed when the Newton iteration appears to be slowly converging" << std::endl;
    os << "  --ida-max-iters-ic=<value>           Maximum number of Newton iterations allowed in any one attempt to solve the initial conditions calculation problem" << std::endl;
    os << "  --ida-line-search-off                Disable the linesearch algorithm" << std::endl;

    os << "  --ida-print-jacobian                 Whether to print the Jacobian matrices while debugging\n";
  }

  void CommandLineOptions::parseCommandLineOptions(const argh::parser& options) const
  {
    options("ida-relative-tolerance", getOptions().relativeTolerance) >> getOptions().relativeTolerance;
    options("ida-absolute-tolerance", getOptions().absoluteTolerance) >> getOptions().absoluteTolerance;
    options("ida-max-algebraic-abs-tol", getOptions().maxAlgebraicAbsoluteTolerance) >> getOptions().maxAlgebraicAbsoluteTolerance;
    options("ida-time-scaling-factor-ic", getOptions().timeScalingFactorInit) >> getOptions().timeScalingFactorInit;

    options("ida-max-steps", getOptions().maxSteps) >> getOptions().maxSteps;
    options("ida-initial-step-size", getOptions().initialStepSize) >> getOptions().initialStepSize;
    options("ida-min-step-size", getOptions().minStepSize) >> getOptions().minStepSize;
    options("ida-max-step-size", getOptions().maxStepSize) >> getOptions().maxStepSize;
    options("ida-max-err-test-fails", getOptions().maxErrTestFails) >> getOptions().maxErrTestFails;
    getOptions().suppressAlg = options["ida-suppress-alg-vars"] ? SUNTRUE : SUNFALSE;
    options("ida-max-nonlin-iters", getOptions().maxNonlinIters) >> getOptions().maxNonlinIters;
    options("ida-max-conv-fails", getOptions().maxConvFails) >> getOptions().maxConvFails;
    options("ida-nonlin-conv-coef", getOptions().nonlinConvCoef) >> getOptions().nonlinConvCoef;
    options("ida-nonlin-conv-coef-ic", getOptions().nonlinConvCoefIC) >> getOptions().nonlinConvCoefIC;
    options("ida-max-steps-ic", getOptions().maxStepsIC) >> getOptions().maxStepsIC;
    options("ida-max-jacs-ic", getOptions().maxNumJacsIC) >> getOptions().maxNumJacsIC;
    options("ida-max-iters-ic", getOptions().maxNumItersIC) >> getOptions().maxNumItersIC;
    getOptions().lineSearchOff = options["ida-line-search-off"] ? SUNTRUE : SUNFALSE;

    getOptions().printJacobian = options["ida-print-jacobian"];
  }
}
