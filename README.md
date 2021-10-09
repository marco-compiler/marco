# MARCO - Modelica Advanced Research COmpiler

*Cheatham's amendment of Conway's Law: If a group of N persons implements a [COBOL] compiler, there will be N-1 passes. Someone in the group has to be the manager.*

## Requirements
### Boost graph
```bash
sudo apt install libboost-all-dev
```
### LLVM & MLIR

MLIR is not currently included in the prebuilt packages, and thus we need to build LLVM from scratch. We also need to select a specific commit, as MLIR is subject to fast changes and MARCO can be possibly not be yet compatible with the latest commits. 
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout d7b669b3a30345cfcdb2fde2af6f48aa4b94845d
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/install_path -DLLVM_USE_LINKER=gold -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;mlir;openmp" ../llvm
make install
```

### LIT
MARCO uses LLVM's LIT to run the regression tests.

```bash
sudo apt install python3-pip
pip3 install lit
```

## Optional requirements
* doxygen to invoke the doc generation target
* genhtml and lcov to get the coverage report.

## Cloning
[google-test](https://github.com/google/googletest/) is used as testing framework. Thus, a simple clone would not be sufficient:

```bash
git clone https://github.com/looms-polimi/marco.git
cd marco
git submodule update --recursive --init
```

## Building
```bash
cd marco
mkdir build && cd build
cmake -DLLVM_DIR=/llvm_install_path/lib/cmake/llvm -DMLIR_DIR=/llvm_install_path/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/lit_executable_path ..
make all
```

## Testing
Regression tests can be run with `make check`

Unit tests can be run with `make test`

## Project structure
The project is divided into tools and libs. Conceptually a lib does something and a tool lets you do the same things from command line.
### Libraries structure
No nested libs are allowed, and each lib must be placed in the ./lib folder. The reason for this choice is because dependencies are expressed with the cmake target_link_libraries command rather than with the position in the tree folder. Furthermore this allows for tools to autodetect which targets are available by just querying the subfolder names in lib. Should the need to visually check the library dependecies ever arise then use a cmake dependecy generator.

A library must be composed of
* A `CMakeLists.txt` file that invokes `marcoAddLibrary`. It must be located in the root of the library: `./lib/libName/CMakeLists.txt`
* An `include` folder. All the headers file in this folder will populate the public interface of this library.
* A `src` folder. This folder can be structured in any way, since it is private. Private headers can be placed here.
* A `test` folder. Read in the test section for further information

#### The marcoAddLibrary macro
This macro behaves like CMake's `add_library`, but it also does the following:
* Specify that the include directory is `pathToLib/include`
* Create an alias for the library called `marco::libName`
* Set the language level to C++17
* Select which and where files will be installed
* Include the tests subfolder

### Executables
As is the case for libraries all executables must be placed in the top level of the tool folder. A tool is composed of the following:
* folder located in ./tool/toolName that contains a CMakeLists.txt that invokes marcoAddTool
* a src folder that must contain a Main.cpp file
* a test folder that can contain anything

### The marcoAddTool macro
The marco add tool macro accepts a name for the tool as first argument and the name of libraries the tool will be depending on. The macro will
* Set the dependencies
* Create an alias called marco::toolName for the tool
* set the language level to c++17
* select what will be installed where
* include the test subdirectory

### Testing
Each library and tool should include a `test` subfolder containing its tests. The `CMakeLists.txt` inside this folder can use the `marcoAddTest` macro. If it does, then the `test` folder must contain a `src` subdirectory with the tests files.

#### The marcoAddTest macro
This macro takes the name of the executable that will contain the tests, it must be the name of the library itself. the postfix test will be added.
all other arguments must be the path to the cpp file that will compose the body of the test.
The macro will
* create the executable
* set were it must be installed.
* set the language level
* set a dependecy to the library with the same name

After building the test can be found in buildfolder/lib/libName/test/libNameTest


## Old README

### DEBUG DEPENDENCIES 
Due to performance reasons llvm behaviour can be a lot different based upon the cmake flags used to compile llvm itself. If you require a debug build of this software read the following.
We raccomend to build the debug version of llvm from sources. Use the monorepo https://github.com/llvm/llvm-project 
Include the clang subproject since currently we are using that for the last stage of compilation. clang tools is not required. 
Do remember to checkout the llvm9 branch rather than building master.

A good talk about buidling llvm is (notice that not all steps are required if your are using the monorepo):
https://www.youtube.com/watch?v=uZI_Qla4pNA

#### -DLLVM_ABI_BREAKING_CHECKS
https://llvm.org/docs/CMake.html . That flags changes the behaviour of llvm::Expected to make it more strict in debug mode and faster in release mode. If you write code in release mode it may work release mode it will look like it is working but in reality errors will be silent and it will break in debug mode. DO NOT WRITE NEW CODE WITHOUT THIS FLAG ENABLED.

### -DLLVM_ENABLE_ASSERTIONS
In llvm preconditions of fuctions are written as assertions. This is because requirements should never be violated in correct code, thus it would wastefull to check them in release mode. On the other hand we do want those checks in debug mode since they usaully provide  a much more informative error message when triggered. Thus, a great deal of functions in llvm begins with the invocatio of assert and a debug build can be up to 20 times slower than a release build.

### Multiple versions of llvm
Notice that keeping separated multiple version of multiples types of llvm can be painfull. A trick is to never install llvm and to always provide the correct version to cmake. 

### Cloning
#### Why is not boost graph a submodule as well?
boost libraries have a special status within cmake. They are not build with cmake but cmake itself provide a way of searching them in the system when they are used as dependecies. They are not as easy as it is for google-test to make them work.

#### Why is not boost llvm a submodule as well?
It would indeed must simplify the compatibility, it is not just for performace reasons.

### Building the release version
```bash
mkdir release 
cd release
cmake ..
make all
```


### Building the debug version
Minimal command
```bash
mkdir build 
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make all
```
Raccomended command
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILE
R=g++ -G Ninja -DCMAKE_EXE_LINKER_FLAGS='-fuse-ld=gold' -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter --coverage
" -DLLVM_DIR=HEEEEEERE --build ../
ninja all
```
#### What does this command does?
* CMAKE_BUILD_TYPE: set the version to debug mode
* CMAKE_EXPORT_COMPILE_COMMANDS: generates the compile_commands.json that contains informations about how compile each cpp file in the project. This file is often required by many tools and editors. In particular is needed by clang-tidy and clang-format. Read later about those tools.
* -G Ninja: select Ninja over make as a build tool. Ninja is faster
* CMAKE_EXE_LINKER_FLAGS="-fuse-ld=gold" select a different linker over ld that is much more performant. lld could be used as well if it is installed with llvm.
* DCMAKE_CXX_FLAGS: Enables to count all warning as werrs excepts for unused variables. we allow for unused variables because clang and gcc have different opinions about what is a unused variable. Furthermore this enables the coverage and allows to generate the coverage report.
* CMAKE_INSTALL_PREFIX selects were the library will be installed, read the installation sections to see more
* LLVM_DIR= the folder in which the finder of llvm will be searched. As an example if llvm is installed in usr/local/lib the correct value is "/usr/local/lib/cmake/llvm" 

### Installation
Once it has been built it can be installed. If you run make install or ninja install depending on your configurations marco will be installed. When running
`cmake ..` you can configure the install location with `-DCMAKE_INSTALL_PREFIX=PATH`. After the installation the header, the libraries and the tools will be deployed in the target folder. Notice that debug libraries have a different name thus they can cohexists  with the release library in the install directory.

### Packaging
If the debug version is located in the prjectRoot/build folder and the release version is located in the projectRoot/release folder and the installation path is ./install for both and they have been build and installed you can create a package. Full command
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=./install
make all
cd ../
mkdir release
cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make all
cd ..
cpack -G ZIP
``` 
It will produce a zip with debug and release libraries headers and tools




## An example of what can be done
Once everything has been built and installed we can try the tools.
Consider the following simple example:
```modelica
model SimpleDer 
  final parameter Real tau = 5.0;
  Real[10] x(start = 0.0);
equation
  tau*der(x[1]) = 1.0;
  for i in 2:10 loop
	tau*der(x[i]) = 2.0*i;
  end for;
end SimpleDer;
```
Each der_x will be twice as much as the previous, thus we expect to see that each x will increase at twice the speed as the previous.

ASTDumper allows as to see how the ast looks like for this model:
```
ASTDumper model.mo
```
It will dump it before the type checking,, so many variables are of unknown type
To see the ast after the typechecking phase we can do

```
omcc -dt model.mo
```

To see what happens after the costant folding phase we can do
```
omcc -df model.mo
```
To see what the intermediate rappresentation looks like 
```
omcc -d model.mo
```
At this point we can save it to file with -o
```
omcc -d model.mo -o file.mod
```
If we are interested to see how it can be causalized we can use modmatch which will produce a dot file with the matched system. There flags to enable and disable some parts of the graph.
```
modmatch file.mod --dumpGraph=graph.dot
```
If we are just interested into the matched model we can do
```
omcc -dm model.mo
```
The left hand of the variables  describe which variable each equation has been matched too.

Now we can schedule the system, we can see the dependency graph with modsched
```
modsched --dumpGraph matched.mo > graph.dot 
```

If we are interested in the code after the matching we can do
```
 omcc -dsched model.mo 
 ```
Finally if we are interested into the llvm ir we can simply do
```
omcc -o out.bc model.mo
```
