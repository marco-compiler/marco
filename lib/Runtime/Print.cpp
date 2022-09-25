#include "marco/Runtime/Print.h"
#include "marco/Runtime/Utils.h"
#include <iomanip>

PrinterConfig& printerConfig()
{
  static PrinterConfig obj;
  return obj;
}

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

#ifdef MARCO_CLI

namespace marco::runtime::formatting
{
  class CommandLineOptions : public cli::Category
  {
    std::string getTitle() const override
    {
      return "Formatting";
    }

    void printCommandLineOptions(std::ostream& os) const override
    {
      os << "  --scientific-notation      Print the values using the scientific notation.\n";
      os << "  --precision=<value>        Set the number of decimals to be printed.\n";
    }

    void parseCommandLineOptions(const argh::parser& options) const override
    {
      printerConfig().scientificNotation = options["scientific-notation"];
      options("precision", printerConfig().precision) >> printerConfig().precision;
    }
  };

  std::unique_ptr<cli::Category> getCLIOptionsCategory()
  {
    return std::make_unique<CommandLineOptions>();
  }
}

#endif

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

#ifdef MARCO_PROFILING

PrintProfiler::PrintProfiler() : Profiler("Simulation data printing")
{
  registerProfiler(*this);
}

void PrintProfiler::reset()
{
  booleanValues.reset();
  integerValues.reset();
  floatValues.reset();
  stringValues.reset();
}

void PrintProfiler::print() const
{
  std::cerr << "Time spent on printing boolean values: " << booleanValues.totalElapsedTime() << " ms\n";
  std::cerr << "Time spent on printing integer values: " << integerValues.totalElapsedTime() << " ms\n";
  std::cerr << "Time spent on printing float values: " << floatValues.totalElapsedTime() << " ms\n";
  std::cerr << "Time spent on printing strings: " << stringValues.totalElapsedTime() << " ms\n";
}

PrintProfiler& printProfiler()
{
  static PrintProfiler obj;
  return obj;
}

#endif

//===----------------------------------------------------------------------===//
// print
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  void print_void(T value)
  {
    std::cout << std::scientific << value << std::endl;
  }

  template<>
  void print_void<bool>(bool value)
  {
    std::cout << std::boolalpha << value << std::endl;
  }
}

RUNTIME_FUNC_DEF(print, void, bool)
RUNTIME_FUNC_DEF(print, void, int32_t)
RUNTIME_FUNC_DEF(print, void, int64_t)
RUNTIME_FUNC_DEF(print, void, float)
RUNTIME_FUNC_DEF(print, void, double)

namespace
{
  template<typename T>
  void print_void(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType memRef(*array);
    std::cout << std::scientific << memRef << std::endl;
  }

  template<>
  void print_void<bool>(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType memRef(*array);
    std::cout << std::boolalpha << memRef << std::endl;
  }
}

RUNTIME_FUNC_DEF(print, void, ARRAY(bool))
RUNTIME_FUNC_DEF(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(float))
RUNTIME_FUNC_DEF(print, void, ARRAY(double))
