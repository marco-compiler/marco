#include <iostream>
#include "serial.h"
#include "registers.h"

//PA2 : USART2 TX
//PA3 : USART2 RX

static const int bufsize=16;
static char rxbuffer[bufsize];
static int putpos=0;
static int getpos=0;
static volatile int numchar=0;

void USART2_IRQHandler()
{
	unsigned int status=USART2->SR; //Read status of usart peripheral
	char c=USART2->DR;              //Read possibly received char
	if(status & USART_SR_RXNE)      //Did we receive a char?
	{
		if(numchar==bufsize) return; //Buffer full
		rxbuffer[putpos]=c;
		if(++putpos >= bufsize) putpos=0;
		numchar++;
	}
}

SerialPort::SerialPort()
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

	GPIOA->AFR[0] |= (7<<(2*4)) | (7<<(3*4));
	GPIOA->MODER |= (2<<(2*2)) | (2<<(3*2));

	USART2->CR1 = USART_CR1_UE | USART_CR1_RXNEIE;
	USART2->BRR = (130<<4) | (3<<0); //19200baud
	USART2->CR1 |= USART_CR1_TE | USART_CR1_RE;

	NVIC->ISER[1]=1<<6;
}

SerialPort::SerialPort(int br){
	int PLLP = (RCC -> PLLCFGR  >> 16) && 0b01 ? 4 : (RCC -> PLLCFGR  >> 16) && 0b00 ? 2 : (RCC -> PLLCFGR  >> 16) && 0b10 ? 6 : 8;
	int sys_frequency =  (((RCC -> CFGR >> 2 ) & 0b11)== 0b10) * 16* ( (RCC -> PLLCFGR >> 6) & 0b111111111) /(RCC -> PLLCFGR & 0b11111) / PLLP;
	int apb_frequency = sys_frequency / 2;
	float div = apb_frequency * 1000000 / (16.0 * br);
	int mantissa = div;
	int frac = (div - mantissa) * 16;
	
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

	GPIOA->AFR[0] |= (7<<(2*4)) | (7<<(3*4));
	GPIOA->MODER |= (2<<(2*2)) | (2<<(3*2));

	USART2->CR1 = USART_CR1_UE | USART_CR1_RXNEIE;
	USART2->BRR = (mantissa<<4) | (frac<<0); 
	USART2->CR1 |= USART_CR1_TE | USART_CR1_RE;

	NVIC->ISER[1]=1<<6;

}	

void SerialPort::write(const char *str)
{
	while((*str)!='\0')
	{
		//Wait until the hardware fifo is ready to accept one char
		while((USART2->SR & USART_SR_TXE)==0) ;
		USART2->DR=*str++;
	}
}

bool SerialPort::available() const
{
	return numchar>0;
}

char SerialPort::read()
{
	//Wait until the interrupt puts one char in the buffer
	while(numchar==0) ;

	asm volatile("cpsid i"); //Disable interrupts
	char result=rxbuffer[getpos];
	if(++getpos >= bufsize) getpos=0;
	numchar--;
	asm volatile("cpsie i"); //Enable interrupts
	return result;
}
