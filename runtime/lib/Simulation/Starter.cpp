extern "C" int runSimulation(int argc, char* argv[]);

// Keeping the 'main' function within a separated library allows the user to
// possibly define his own entry point for the simulation.

int main(int argc, char* argv[])
{
  return runSimulation(argc, argv);
}
