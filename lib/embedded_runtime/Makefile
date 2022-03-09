CC  := arm-miosix-eabi-gcc
CXX := arm-miosix-eabi-g++
AS  := arm-miosix-eabi-as
CP  := arm-miosix-eabi-objcopy

CPU := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16

AFLAGS   := $(CPU) 
CFLAGS   := $(CPU) -O0 -g -c -I./include/marco/driver/CMSIS/Include -I./include/marco/driver/CMSIS/Device/ST/STM32F4xx/Include -I./include/marco/driver
CXXFLAGS := $(CPU) -O0 -fno-exceptions -fno-rtti -g -c -I./include/marco/driver/CMSIS/Include -I./include/marco/driver/CMSIS/Device/ST/STM32F4xx/Include -I./include/marco/driver -I../../include/public
LFLAGS   := $(CPU) -Wl,-T./src/linker.ld,-Map,./test/main.map #-nostdlib


SRC := ./src/startup.s ./test/main.cpp ./src/serial.cpp ./src/pll_driver.cpp ./src/heap.cpp

SRCS :=  ./src/startup.s ./src/init.cpp ./src/serial.cpp ./src/pll_driver.cpp ./src/heap.cpp ./src/stub.cpp
MLIB := ../runtime/BuiltInFunctions.cpp ../runtime/Math.cpp ../runtime/MemoryManagement.cpp ../runtime/Profiling.cpp ../runtime/Runtime.cpp ../runtime/UtilityFunctions.cpp

OBJ := $(addsuffix .o, $(basename $(SRC))) 

OBJLIB :=  ./staticlib/BuiltInFunctions.o ./staticlib/Math.o ./staticlib/MemoryManagement.o ./staticlib/Profiling.o ./staticlib/Runtime.o ./staticlib/UtilityFunctions.o
OBJS := $(addsuffix .o, $(basename $(SRCS))) ./test/nomain.o

#-L ./ -l:libMARCORuntime.a
#arm-miosix-eabi-ld -r -o output.o ./libMARCORuntime.a ./test/nomain.o $(OBJS) 
#$(CP) -O binary init.elf init.bin
all: $(OBJ)
	$(CXX) $(LFLAGS) -o main.elf $(OBJ)
	$(CP) -O binary main.elf main.bin

sim: $(OBJS)
	$(CXX) $(LFLAGS) -o init.elf $(OBJS) -L. -lMARCORuntime 
	$(CP) -O binary init.elf init.bin


lib: $(MLIB)
	mkdir staticlib
	$(CXX) $(CXXFLAGS) ../runtime/BuiltInFunctions.cpp -o ./staticlib/BuiltInFunctions.o
	$(CXX) $(CXXFLAGS) ../runtime/Math.cpp -o ./staticlib/Math.o
	$(CXX) $(CXXFLAGS) ../runtime/MemoryManagement.cpp -o ./staticlib/MemoryManagement.o
	$(CXX) $(CXXFLAGS) ../runtime/Profiling.cpp -o ./staticlib/Profiling.o
	$(CXX) $(CXXFLAGS) ../runtime/Runtime.cpp -o ./staticlib/Runtime.o
	$(CXX) $(CXXFLAGS) ../runtime/UtilityFunctions.cpp -o ./staticlib/UtilityFunctions.o
	arm-miosix-eabi-ar -rcs libMARCORuntime.a $(OBJLIB)
clean:
	-rm $(OBJ) main.elf main.bin main.map

cleansim:
	rm $(addsuffix .o, $(basename $(SRCS))) init.bin init.elf init.map

cleanlib:
	rm -r staticlib libMARCORuntime.a
	

%.o : %.s
	$(AS) $(AFLAGS) $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ 
