#include "serial.h"
#include "registers.h"

//PA2 : USART2 TX
//PA3 : USART2 RX
#define BUF_SIZE 32
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

void SerialPort::write(const char c){
	while((USART2->SR & USART_SR_TXE)==0) ;
	USART2->DR = c;
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

void SerialPort::write(const int* n){
	char s[64];
	tochar(*n,s);
	this->write(s);
}


void SerialPort::write(const int n){
	char s[64];
	tochar(n,s);
	this->write(s);
}

void SerialPort::write(long long int n){
	int m = (int) n;
	this->write(m);
}

void SerialPort::write(long int n){
	int m = (int) n;
	this->write(m);
}

void SerialPort::write(double n){
	float m = (float) n;
	this->write(m);
}



void SerialPort::write(float f, const int p){
	char s[BUF_SIZE] = {0};
	tochar(f,p,s);
	this->write(s);
}

void SerialPort::write(const float f){
	this->write(f,2);
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


char* SerialPort::tochar(int i, char* res){
	

   int len = 0;
   for(; i >0; ++len)
   {	
      res[len] = i%10+'0';
      i/=10; 
   }
   //res[len++] = '\r';
   //res[len++] = '\n';
   res[len] = 0; //null-terminating

   //now we need to reverse res
   for(int i = 0; i < len/2; ++i)
   {
       char c = res[i]; res[i] = res[len-i-1]; res[len-i-1] = c;
   }
   return res;
   
};

char* SerialPort::tochar(long int i, char* res){
	

   int len = 0;
   for(; i >0; ++len)
   {	
      res[len] = i%10+'0';
      i/=10; 
   }
   //res[len++] = '\r';
   //res[len++] = '\n';
   res[len] = 0; //null-terminating

   //now we need to reverse res
   for(int i = 0; i < len/2; ++i)
   {
       char c = res[i]; res[i] = res[len-i-1]; res[len-i-1] = c;
   }
   return res;
   
};


char* SerialPort::tochar(const float x,const int precision,char* p){
	int a,b,c,k,l=0,m,i=0,j;
	float f = x;
	int o = 0;
	// check for negetive float
	if(f<0.0)
	{
		
		p[i++]='-';
		f*=-1;
		o++;
	}
	
	a=f;	// extracting whole number
	f-=a;	// extracting decimal part
	k = precision;
	
	// number of digits in whole number
	do
	{
		l = power(10,k);
		m = a/l;
		if(m>0)
		{
			break;
		}
	k--;
	}while(k>-1);

	// number of digits in whole number are k+1
	for(; a>0; ++i){
   		{
      		p[i] = a%10+'0';
      		a/=10; 
   		}
	}

   
   //now we need to reverse res
   for(; o < i/2; ++o)
   {
       char c = p[o]; p[o] = p[i-o-1]; p[i-o-1] = c;
   }
	p[i++] = '.';
	

	for(l=0;l<precision;l++)
	{
		f*=10.0;
		b = f;
		//write(b);
		p[i++]=b+48;
		f-=b ;
	}
	if( f* 10 >= 5) p[i-1] = p[i-1] + 1;
	p[i]='\0';
}

int SerialPort::power(const int n,const int p){
	return p == 0 ? 1 : n*power(n,p-1);
}